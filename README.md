# 🪢 CKnot (Chinese Knot Orchestrator)

**CKnot** is a cutting-edge agentic workflow orchestration framework built with Python, LangGraph, and FastAPI. Inspired by the intricate and interconnected nature of a Chinese Knot, the project coordinates multiple specialized agents under a central "Boss" orchestrator to handle complex system debugging and general assistance tasks.

## ✨ Key Features

-   **Hierarchical Multi-Agent System**: A central "Boss Agent" (`cknot`) delegates tasks to specialized agents like the `Deep Search`, `Log Parser` and `Code Fixer`.
-   **Interactive Rich CLI**: A professional terminal interface featuring Markdown rendering, live status spinners, and a hierarchical slash-command registry (`/llms`, `/agents`, `/status`, etc.).
-   **Persistent State Management**: Full conversation persistence using Redis-backed LangGraph checkpointing, allowing workflows to survive restarts and handle human-in-the-loop interruptions.
-   **Dynamic LLM Service Registry**: Register, enable, and switch between multiple LLM providers (OpenAI, local vLLM, etc.) at runtime without code changes.
-   **Production-Grade Logging**: Sophisticated logging featuring sensitive data redaction (API keys/passwords) and dynamic routing of logs to user-specific files.
-   **Telemetry & Cost Tracking**: Built-in tracking of token consumption and estimated costs per agent turn.
-   **Modern Tooling**: Powered by `uv` for lightning-fast dependency management and containerized with Docker/Podman support.

## 🚀 Getting Started

### Prerequisites
- Docker or Podman
- `uv` (recommended for local development)

### Quick Start

1.  **Initialize the project**:
    ```bash
    ./bootstrap.sh init
    ```

2.  **Configure environment**:
    Update the `.env` file in the root with your LLM provider details and Redis configuration.

3.  **Build and start the API server**:
    ```bash
    ./bootstrap.sh build
    ./bootstrap.sh start
    ```

4.  **Enter the Interactive CLI**:
    ```bash
    ./bootstrap.sh cli
    ```

## 🛠 CLI Slash Commands

The interactive CLI supports a hierarchical command system. Type `/help` inside the CLI turn to see all available options:

```
   ____ _  __ _   _  ___ _____ 
  / ___| |/ /| \ | |/ _ \_   _|
 | |   | ' / |  \| | | | || |  
 | |___| . \ | |\  | |_| || |  
  \____|_|\_\|_| \_|\___/ |_|  
version 0.0.1-alpha

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
cknot Interactive CLI (Session: cli_session_1779982985)
Type /exit or /quit to end the session.

───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
You > search for the latest advancements and breakthroughs in AI technology today
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
I will initiate a deep search to find the most recent news and technical breakthroughs in the field of Artificial          
Intelligence.                                                                                                              

TRIGGER_DEEP_SEARCH: latest advancements and breakthroughs in AI technology 2023 2024                                      
╭──────────────────────────────────────────────────── Action Required ────────────────────────────────────────────────────╮
│ The agent is requesting to execute: ('deep_search',)                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Authorize execution? (yes/no): yes
╭──────────────────────────────────────────────────── Action Required ────────────────────────────────────────────────────╮
│ The agent is requesting to execute: ('tools',)                                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Authorize execution? (yes/no): yes
The previous search did not yield relevant news results regarding AI advancements. I will re-run the search with more      
specific queries to find recent breakthroughs and developments.                                                            

TRIGGER_DEEP_SEARCH: AI news developments today, new large language models released in 2024, latest breakthrough in        
artificial intelligence May 2024                                                                                           
╭──────────────────────────────────────────────────── Action Required ────────────────────────────────────────────────────╮
│ The agent is requesting to execute: ('tools',)                                                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Authorize execution? (yes/no): no 

✘ Execution denied. Returning to chat loop.
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
You > /agents
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────── Active Agents in Graph ──────────────────────────────────────────────────
- cknot        LLM: default-llm (default) → Boss Orchestrator
- log_parser   LLM: default-llm (default) → Log Analysis Specialist
- code_fixer   LLM: default-llm (default) → Remediation specialist
- deep_search  LLM: default-llm (default) → Deep Research & Analysis
- tools        LLM: default-llm (default) → Tool Execution Engine
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
You > /llms
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
                             Registered LLM Services                              
┏━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ ID          ┃ Provider ┃ Model              ┃ Status  ┃ Valid ┃ Usage (In/Out) ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ local-vllm  │ vllm     │ Qwen/Qwen3-0.6B    │ ENABLED │   ✔   │          0 / 0 │
│ default-llm │ ollama   │ my-qwen-3.6:latest │ ENABLED │   ✔   │    2449 / 2028 │
└─────────────┴──────────┴────────────────────┴─────────┴───────┴────────────────┘
───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
You > /llms
             list    Lists all registered LLM services and their health status.  
             add     Interactively registers a new LLM service.
             rm      Removes an LLM service by ID.               
             test    Runs a connectivity check for an LLM service.
             load    Loads LLM services from a JSON or YAML file.
             enable  Enables an LLM service.   
```

## 🛡 Security & Reliability

-   **Data Privacy**: The `RedactingFilter` automatically masks sensitive patterns like `api_key` or `password` in all system logs.
-   **State Isolation**: Conversations are isolated via `session_id`, ensuring multi-tenant safety.
-   **Health Checks**: The system performs pre-flight checks on Redis and LLM providers to ensure service availability before starting workflows.

## API Tests
1. Access the UI: Start your API server (./bootstrap.sh start) and navigate to http://localhost:9999/docs in your browser.

2. Authorize: Click the Authorize button at the top right. Use the /token endpoint credentials to unlock protected endpoints (indicated by the lock icon).

3. Test Orchestration:
   - Use the POST /chat endpoint to send a message like "Search for latest AI news".
   - If the response returns requires_action: true and next_node: "deep_search", use the POST /approve/{session_id} endpoint to authorize the specialist to run.

4. Audit Infrastructure: Expand the llms, tools, and users sections to review current system configurations, check token usage, or manage user profiles.

![Swagger UI APIs](./docs/images/swaggeruiapis.png)
![Agents List API](./docs/images/agentslistapi.png)

---
*Developed with passion for robust agentic orchestration.*
---