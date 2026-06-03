import logging
import os
from typing import Type, Any, ClassVar
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from cknot.utils.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="The natural language question to ask the knowledge base.")

class LlamaIndexRetrieverTool(BaseTool):
    """
    A tool that uses LlamaIndex to query a local directory of documents.
    """
    name: str = "knowledge_base"
    description: str = "Search through internal documentation and system manuals for specific technical answers."
    args_schema: Type[BaseModel] = KnowledgeBaseInput
    default_source_path: ClassVar[str] = "/app/ragsource"
    default_store_path: ClassVar[str] = "/app/ragstore"
    
    data_path: str = Field(default="/app/ragsource")
    storage_dir: str = Field(default="/app/ragstore")
    service_id: str = Field(default="default-llm")
    embed_service_id: str = Field(default="default-embedding")
    _index: Any = PrivateAttr()

    def __init__(
        self,
        data_path: str = "/app/ragsource",
        storage_dir: str = "/app/ragstore",
        service_id: str = "default-llm",
        embed_service_id: str = "default-embedding"
    ):
        logger.debug(f"Initing LlamaIndexRetrieverTool with {data_path=}, {storage_dir=}, {service_id=}, {embed_service_id=}")
        # Resolve relative paths using default source/store locations
        if not os.path.isabs(data_path) and not os.path.exists(data_path):
            # Only prepend default path if the file doesn't exist relative to CWD
            potential_path = os.path.join(self.default_source_path, data_path)
            data_path = potential_path if os.path.exists(potential_path) else data_path
            
        if not os.path.isabs(storage_dir):
            storage_dir = os.path.join(self.default_store_path, storage_dir)

        # Initialize the Pydantic model with the resolved fields
        super().__init__(
            data_path=data_path,
            storage_dir=storage_dir,
            service_id=service_id,
            embed_service_id=embed_service_id
        )
        
        # Fetch the shared embedding instance from LLMManager
        embed_model = LLMManager().get_llama_index_embeddings(self.embed_service_id)
        self._index = self._get_or_create_index(embed_model)

    def _get_or_create_index(self, embed_model):
        # Ensure the directory exists
        if not os.path.exists(self.storage_dir):
            logger.debug(f"RAG Indexing from {self.data_path} to {self.storage_dir}")
            # Handle case where data_path is a specific file or a directory
            if os.path.isfile(self.data_path):
                documents = SimpleDirectoryReader(input_files=[self.data_path]).load_data()
            else:
                documents = SimpleDirectoryReader(self.data_path).load_data()
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            index.storage_context.persist(persist_dir=self.storage_dir)
            return index
        else:
            logger.debug(f"RAG Loading index from {self.storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            return load_index_from_storage(storage_context, embed_model=embed_model)

    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        logger.debug(f"RAG Sync Quering: {query}")
        llm = LLMManager().get_llama_index_llm(self.service_id)
        query_engine = self._index.as_query_engine(llm=llm)
        response = query_engine.query(query)
        logger.debug(f"RAG Sync Query response: {response}")
        return str(response)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        logger.debug(f"RAG Async Quering: {query}")
        llm_manager = LLMManager()
        llm = llm_manager.get_llama_index_llm(self.service_id)

        # LlamaIndex query engines can be converted to async
        query_engine = self._index.as_query_engine(llm=llm, streaming=False)
        # Simple wrap for demonstration; LlamaIndex has native aquery support
        response = await query_engine.aquery(query)
        logger.debug(f"RAG Async Query response: {response}")
        return str(response)