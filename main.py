import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict

import openai
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent


servers_config: Dict[str, dict] = {
    "DappierServer": {
        "transport": "sse",
        "url": "http://103.116.38.80:3003/sse"
    },
    "ArxivServer": {
        "transport": "sse",
        "url": "http://103.116.38.80:3000/sse"
    },
    "ExcelServer": {
        "transport": "sse",
        "url": "http://103.116.38.80:3002/sse"
    }
}

class ServerRegistration(BaseModel):
    name: str
    url: str

class AskPayload(BaseModel):
    question: str

async def init_persistent_client(config: Dict[str, dict]) -> MultiServerMCPClient:
    """
    Helper function to initialize the persistent MCP client.
    """
    client = MultiServerMCPClient(config)
    await client.__aenter__()
    return client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager to set up and tear down the persistent client.
    """
    app.state.persistent_client = await init_persistent_client(servers_config)
    try:
        yield
    finally:
        await app.state.persistent_client.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/api/servers")
def list_servers():
    return servers_config

@app.post("/api/servers")
async def register_server(reg: ServerRegistration):
    """
    Register a new server and reinitialize the persistent client.
    """
    # Update the global configuration.
    servers_config[reg.name] = {"transport": "sse", "url": reg.url}
    # Reinitialize the persistent client with the updated configuration.
    if hasattr(app.state, "persistent_client"):
        await app.state.persistent_client.__aexit__(None, None, None)
    app.state.persistent_client = await init_persistent_client(servers_config)
    return {"status": "ok"}

class ChatGPTModel:
    """
    A simple wrapper for OpenAI's ChatCompletion API that implements an async callable interface.
    """
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model

    async def __call__(self, messages: dict) -> dict:
        # Use asyncio.to_thread to call the synchronous OpenAI API without blocking the event loop.
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": messages.get("messages", "")}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        return {"messages": answer}

@app.post("/api/ask")
async def ask_question(payload: AskPayload):
    """
    Handle the ask endpoint by invoking the agent using ChatGPT.
    """
    model = ChatGPTModel(model="gpt-3.5-turbo")
    try:
        tools = app.state.persistent_client.get_tools() if servers_config else []
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke({"messages": payload.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during agent invocation: {e}")
    
    answer_text = response.get("messages", "No answer received.")
    return {"answer": answer_text}
