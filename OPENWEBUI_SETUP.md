# Open WebUI + BoardDocs RAG Setup Guide

This guide configures Open WebUI to use a local Qwen3 model as a conversational
layer with a BoardDocs RAG tool for factual retrieval from Kent School District
public records.

## Architecture

```
User → Open WebUI (port 3000) → Qwen3 via llama-server (port 8080)
                                       ↓ (tool call)
                                 RAG API (port 8000) → Qdrant + Claude
```

The Qwen3 model handles conversation. When it needs factual information about
Kent School District, it calls the BoardDocs RAG tool, which queries the
document corpus and returns cited answers.

## Prerequisites

All three services must be running:

| Service | Port | Check command |
|---------|------|---------------|
| Open WebUI | 3000 | `curl http://localhost:3000` |
| llama-server | 8080 | `curl http://127.0.0.1:8080/v1/models` |
| RAG API | 8000 | `curl http://127.0.0.1:8000/health` |

### Starting services

```bash
# Open WebUI (already running as Podman container)
podman start open-webui

# llama-server (Qwen3-30B-A3B)
systemctl --user start ksd-llama-server

# RAG API (needs ANTHROPIC_API_KEY)
cd ~/ksd-boarddocs-rag/rag_api
set -a && source ~/ksd-boarddocs-rag/.env && set +a
source venv/bin/activate
uvicorn rag_api.main:app --host 127.0.0.1 --port 8000 &
```

## Step 1: Connect llama-server to Open WebUI

1. Open http://localhost:3000 in your browser
2. Click your profile icon (bottom-left) → **Admin Panel**
3. Go to **Settings** → **Connections**
4. Under **OpenAI API**, click **+** to add a new connection:
   - **URL**: `http://host.containers.internal:8080`
   - **API Key**: `none` (any non-empty string works)
5. Click the **verify** button (checkmark icon). It should turn green and show the model `Qwen3-30B-A3B-UD-Q8_K_XL.gguf`
6. Click **Save**

## Step 2: Install the BoardDocs RAG Tool

1. In the Open WebUI sidebar, click **Workspace**
2. Click **Tools** (wrench icon tab)
3. Click the **+** button to create a new tool
4. Set the tool name to: `BoardDocs RAG Search`
5. Set the description to: `Search Kent School District BoardDocs documents`
6. In the code editor, **paste the entire contents** of:
   ```
   ~/ksd-boarddocs-rag/boarddocs_rag_tool.py
   ```
7. Click **Save**

### Configure Tool Valves (if needed)

After saving, click the gear icon on the tool to adjust settings:

| Valve | Default | Notes |
|-------|---------|-------|
| RAG_API_URL | `http://host.containers.internal:8000` | Correct for Podman containers |
| TENANT_ID | `kent_sd` | Only change for multi-district deployments |
| TOP_K | `10` | Number of document chunks to retrieve |
| TIMEOUT_SECONDS | `60` | Increase if queries consistently time out |

The default RAG_API_URL uses `host.containers.internal` which resolves to the
host machine from inside any Podman container on Linux. Do not change this to
`localhost` — it won't work from inside the container.

## Step 3: Create a Model Preset with the Tool Enabled

This step creates a named model that has the BoardDocs tool permanently enabled,
so users don't need to manually enable it each conversation.

1. In the Open WebUI sidebar, click **Workspace**
2. Click **Models** tab
3. Click **+** to create a new model
4. Configure:
   - **Name**: `BoardDocs Assistant`
   - **Model ID**: `boarddocs-assistant`
   - **Base Model**: Select `Qwen3-30B-A3B-UD-Q8_K_XL.gguf` from the dropdown
   - **System Prompt**:
     ```
     You are a helpful research assistant for Kent School District public records.
     When the user asks about school board meetings, district policies, budgets,
     personnel matters, or any official district business, use the search_boarddocs
     tool to retrieve factual information with citations. Always cite your sources
     using the [Source N] references provided by the tool. For general questions
     not related to Kent School District, answer from your own knowledge.
     ```
   - **Tools**: Enable **BoardDocs RAG Search**
   - **Advanced params → Max Tokens**: Set to `4096`
5. Click **Save**

## Step 4: Verify Everything Works

1. Start a **New Chat** in Open WebUI
2. Select **BoardDocs Assistant** as the model
3. Ask: **"Which Kent School District board meetings discussed the budget most extensively?"**
4. You should see:
   - A brief "Searching BoardDocs..." status indicator
   - A conversational answer from the Qwen3 model
   - Source citations referencing actual BoardDocs documents with dates and URLs
5. Follow up: **"What specific dollar amounts were mentioned?"**
   - The model should call the tool again with a refined budget-related query
   - Response should include specific figures from the documents

## Troubleshooting

### Tool not appearing in model configuration
Make sure the tool was saved without syntax errors. Check the Open WebUI logs:
```bash
podman logs open-webui --tail 20
```

### "Error: Could not connect to the BoardDocs RAG API"
The RAG API is not running or not reachable from the container:
```bash
# Check RAG API is running
curl http://127.0.0.1:8000/health

# Check container can reach it
podman exec open-webui curl -s http://host.containers.internal:8000/health
```

### Model not generating tool calls
Qwen3 uses a reasoning/thinking step before responding. If it's not calling the
tool, the system prompt may need reinforcement. Also verify that:
- The tool is enabled in the model preset (Step 3)
- The tool shows a green status in the chat interface
- Try being explicit: "Search BoardDocs for budget discussions"

### llama-server crashed or slow
```bash
# Check status
systemctl --user status ksd-llama-server

# View logs
journalctl --user -u ksd-llama-server --no-pager -n 50

# Restart
systemctl --user restart ksd-llama-server
```

### Responses are empty or cut off
Increase max_tokens in the model preset's advanced parameters. Qwen3 uses
reasoning tokens internally, so set max_tokens to at least 4096.
