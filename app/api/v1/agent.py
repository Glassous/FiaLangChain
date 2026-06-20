import json
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.security import verify_token
from app.agents import LangGraphAgentRunner

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ModelConfig(BaseModel):
    model: str
    temperature: float = 0.7
    api_key: Optional[str] = ""
    base_url: Optional[str] = ""
    bocha_api_key: Optional[str] = ""
    tavily_api_key: Optional[str] = ""

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    system_prompt: Optional[str] = ""
    tools: List[str] = Field(default_factory=list)
    model_config_data: ModelConfig = Field(alias="model_config")

@router.post("/chat")
async def chat(
    request: ChatRequest,
    token: str = Security(verify_token)
):
    # Initialize the runner
    runner = LangGraphAgentRunner(
        api_key=request.model_config_data.api_key,
        base_url=request.model_config_data.base_url,
        model=request.model_config_data.model,
        temperature=request.model_config_data.temperature,
        bocha_api_key=request.model_config_data.bocha_api_key,
        tavily_api_key=request.model_config_data.tavily_api_key
    )

    async def sse_generator():
        event_queue = asyncio.Queue()
        
        # Start agent runner in the background
        task = asyncio.create_task(
            runner.run(
                messages=[m.model_dump() for m in request.messages],
                system_prompt=request.system_prompt,
                enabled_tools=request.tools,
                event_queue=event_queue
            )
        )

        try:
            while True:
                # Wait for events from the agent runner
                event = await event_queue.get()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                
                if event.get("type") in ("done", "error"):
                    break
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            await task

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream"
    )
