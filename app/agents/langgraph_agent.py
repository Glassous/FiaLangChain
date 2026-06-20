import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.core.config import settings
from app.tools import TOOL_MAP

class LangGraphAgentRunner:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.7, bocha_api_key: str = "", tavily_api_key: str = ""):
        self.llm = ChatOpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=base_url or settings.OPENAI_BASE_URL,
            model=model,
            temperature=temperature,
            streaming=True
        )
        self.bocha_api_key = bocha_api_key
        self.tavily_api_key = tavily_api_key

    async def run(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        enabled_tools: List[str],
        event_queue: asyncio.Queue
    ):
        """Runs the agent loop and puts events into the event_queue."""
        try:
            # 1. Send Start Event
            await event_queue.put({"type": "start"})

            # Convert input messages to LangChain format
            lc_messages = []
            if system_prompt:
                lc_messages.append(SystemMessage(content=system_prompt))
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))

            # 2. Plan Generation Phase (if tools are enabled)
            plan = None
            active_tools = [TOOL_MAP[name] for name in enabled_tools if name in TOOL_MAP]
            
            if active_tools:
                plan = await self._generate_plan(lc_messages, active_tools)
                if plan and plan.get("items"):
                    await event_queue.put({
                        "type": "agent_plan",
                        "data": plan["items"]
                    })

            # Bind tools to LLM
            llm_with_tools = self.llm
            if active_tools:
                llm_with_tools = self.llm.bind_tools(active_tools)

            # 3. Execution Loop
            max_iterations = 10
            iteration = 0
            plan_index = 0

            while iteration < max_iterations:
                iteration += 1

                # If we have a plan, send plan progress updates
                if plan and plan.get("items") and plan_index < len(plan["items"]):
                    await event_queue.put({
                        "type": "plan_item",
                        "data": {"index": plan_index, "status": "in_progress"}
                    })

                # Stream reasoning and tokens from LLM
                response_message = None
                tool_calls = []
                
                async for chunk in llm_with_tools.astream(lc_messages):
                    if chunk.content:
                        # Output ordinary token
                        await event_queue.put({
                            "type": "token",
                            "content": chunk.content
                        })
                    
                    # Gather reasoning chunk if model provides it (e.g. DeepSeek thinking)
                    # Some OpenAI-compatible models send reasoning content in additional fields
                    if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                        reasoning = chunk.additional_kwargs["reasoning_content"]
                        if reasoning:
                            await event_queue.put({
                                "type": "reasoning",
                                "content": reasoning
                            })
                    elif hasattr(chunk, "invalid_tool_calls") or hasattr(chunk, "tool_calls"):
                        # Accumulate tool calls
                        pass
                    
                    if response_message is None:
                        response_message = chunk
                    else:
                        response_message += chunk

                lc_messages.append(response_message)

                # Check for tool calls
                if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                    tool_calls = response_message.tool_calls
                else:
                    tool_calls = []

                if not tool_calls:
                    # Final response completed, exit loop
                    if plan and plan.get("items") and plan_index < len(plan["items"]):
                        # Complete remaining plan items if any
                        for i in range(plan_index, len(plan["items"])):
                            await event_queue.put({
                                "type": "plan_item",
                                "data": {"index": i, "status": "completed"}
                            })
                    break

                # Execute all requested tool calls in parallel/sequence
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    call_id = tc["id"]

                    tool_obj = TOOL_MAP.get(name)
                    if not tool_obj:
                        err_msg = f"Tool '{name}' not found."
                        await event_queue.put({
                            "type": "agent_step",
                            "data": {
                                "index": iteration,
                                "tool_name": name,
                                "tool_input": json.dumps(args, ensure_ascii=False),
                                "tool_output": "",
                                "err": err_msg,
                                "plan_index": plan_index if plan else None
                            }
                        })
                        lc_messages.append(ToolMessage(content=err_msg, tool_call_id=call_id))
                        continue

                    # Invoke the tool
                    try:
                        tool_res = await tool_obj.ainvoke(
                            args,
                            config={"configurable": {"bocha_api_key": self.bocha_api_key, "tavily_api_key": self.tavily_api_key}}
                        )
                        output_str = json.dumps(tool_res, ensure_ascii=False)
                        await event_queue.put({
                            "type": "agent_step",
                            "data": {
                                "index": iteration,
                                "tool_name": name,
                                "tool_input": json.dumps(args, ensure_ascii=False),
                                "tool_output": output_str,
                                "err": "",
                                "plan_index": plan_index if plan else None
                            }
                        })
                        lc_messages.append(ToolMessage(content=output_str, tool_call_id=call_id))
                    except Exception as e:
                        err_str = str(e)
                        await event_queue.put({
                            "type": "agent_step",
                            "data": {
                                "index": iteration,
                                "tool_name": name,
                                "tool_input": json.dumps(args, ensure_ascii=False),
                                "tool_output": "",
                                "err": err_str,
                                "plan_index": plan_index if plan else None
                            }
                        })
                        lc_messages.append(ToolMessage(content=err_str, tool_call_id=call_id))

                # Mark plan item as completed
                if plan and plan.get("items") and plan_index < len(plan["items"]):
                    await event_queue.put({
                        "type": "plan_item",
                        "data": {"index": plan_index, "status": "completed"}
                    })
                    plan_index += 1

            # 4. Finish Event
            await event_queue.put({"type": "done"})

        except Exception as e:
            await event_queue.put({"type": "error", "content": f"Agent failed: {str(e)}"})

    async def _generate_plan(self, messages: List[Any], tools: List[Any]) -> Optional[Dict[str, Any]]:
        """Generates a step-by-step execution plan."""
        try:
            tool_descs = [{"name": t.name, "description": t.description} for t in tools]
            prompt = (
                "你是一个任务规划助手。根据用户的问题和以下可用工具，制定一个执行计划。\n\n"
                f"可用工具：\n{json.dumps(tool_descs, ensure_ascii=False)}\n\n"
                "请以 JSON 格式返回执行计划，格式如下：\n"
                '{"items": [{"id": 1, "description": "步骤描述", "tool_name": "工具名"}, ...]}\n\n'
                '如果不需要任何工具，返回空列表：{"items": []}\n'
                "只返回 JSON，不要包含其他解释或 MarkDown 代码块。"
            )
            
            # Form plan messages context
            plan_msgs = [SystemMessage(content=prompt)]
            # Append last user message
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    plan_msgs.append(msg)
                    break
            
            res = await self.llm.ainvoke(plan_msgs)
            content = res.content.strip()
            
            # Clean JSON markdown blocks if any
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()
                
            return json.loads(content)
        except Exception:
            return None
