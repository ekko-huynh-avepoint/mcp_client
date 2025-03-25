import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Predefined server configuration
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    persistent_client = MultiServerMCPClient(servers_config)
    await persistent_client.__aenter__()
    app.state.persistent_client = persistent_client
    yield
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
    servers_config[reg.name] = {"transport": "sse", "url": reg.url}
    if hasattr(app.state, "persistent_client"):
        await app.state.persistent_client.__aexit__(None, None, None)
    new_client = MultiServerMCPClient(servers_config)
    await new_client.__aenter__()
    app.state.persistent_client = new_client
    return {"status": "ok"}

@app.post("/api/ask")
async def ask_question(payload: AskPayload):
    model = ChatOpenAI(model="gpt-4o-mini")
    try:
        persistent_client = app.state.persistent_client
        tools = persistent_client.get_tools() if servers_config else []
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke({"messages": payload.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during agent invocation: {str(e)}")
    answer_text = response.get("messages", "No answer received.")
    return {"answer": answer_text}