import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from cknot.utils.llm_manager import LLMManager

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="The natural language question to ask the knowledge base.")

class LlamaIndexRetrieverTool(BaseTool):
    """
    A tool that uses LlamaIndex to query a local directory of documents.
    """
    name: str = "knowledge_base"
    description: str = "Search through internal documentation and system manuals for specific technical answers."
    args_schema: Type[BaseModel] = KnowledgeBaseInput
    
    def __init__(self, data_path: str, storage_dir: str = "./storage", service_id: str = "default-llm", embed_service_id: str = "default-embed"):
        super().__init__()
        self.data_path = data_path
        self.storage_dir = storage_dir
        self.service_id = service_id
        self.embed_service_id = embed_service_id
        
        # Fetch the shared embedding instance from LLMManager
        embed_model = LLMManager().get_llama_index_embeddings(self.embed_service_id)
        self._index = self._get_or_create_index(embed_model)

    def _get_or_create_index(self, embed_model):
        # Ensure the directory exists
        if not os.path.exists(self.storage_dir):
            documents = SimpleDirectoryReader(self.data_path).load_data()
            index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
            index.storage_context.persist(persist_dir=self.storage_dir)
            return index
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            return load_index_from_storage(storage_context, embed_model=embed_model)

    def _run(self, query: str) -> str:
        """Use the tool synchronously."""
        llm = LLMManager().get_llama_index_llm(self.service_id)
        query_engine = self._index.as_query_engine(llm=llm)
        response = query_engine.query(query)
        return str(response)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        llm_manager = LLMManager()
        llm = llm_manager.get_llama_index_llm(self.service_id)

        # LlamaIndex query engines can be converted to async
        query_engine = self._index.as_query_engine(llm=llm, streaming=False)
        # Simple wrap for demonstration; LlamaIndex has native aquery support
        response = await query_engine.aquery(query)
        return str(response)