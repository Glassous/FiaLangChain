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
        async def put_event(event: dict):
            # Print event to stdout for docker log tracing
            content_preview = str(event.get("content", ""))[:100]
            print(f"[AgentTrace] Event: {event.get('type')} | Content: {content_preview}", flush=True)
            await event_queue.put(event)

        try:
            # 1. Send Start Event
            await put_event({"type": "start"})

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
            is_single_search = False
            
            if active_tools:
                plan = await self._generate_plan(lc_messages, active_tools)
                if plan and plan.get("items"):
                    items = plan["items"]
                    # If there's only one item and it's web_search, decouple and run as single search
                    if len(items) == 1 and items[0].get("tool_name") == "web_search":
                        is_single_search = True
                    
                    if not is_single_search:
                        await put_event({
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
                    if not is_single_search:
                        await put_event({
                            "type": "plan_item",
                            "data": {"index": plan_index, "status": "in_progress"}
                        })

                # Stream reasoning and tokens from LLM
                response_message = None
                tool_calls = []
                
                async for chunk in llm_with_tools.astream(lc_messages):
                    if chunk.content:
                        # Output ordinary token
                        await put_event({
                            "type": "token",
                            "content": chunk.content
                        })
                    
                    # Gather reasoning chunk if model provides it (e.g. DeepSeek thinking)
                    # Some OpenAI-compatible models send reasoning content in additional fields
                    if hasattr(chunk, "additional_kwargs") and "reasoning_content" in chunk.additional_kwargs:
                        reasoning = chunk.additional_kwargs["reasoning_content"]
                        if reasoning:
                            await put_event({
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
                            await put_event({
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
                        await put_event({
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
                        if is_single_search and name == "web_search":
                            # Decouple: send search results XML tag to client in token stream
                            search_data = {
                                "query": args.get("query", ""),
                                "results": tool_res.get("results", []) if isinstance(tool_res, dict) else [],
                                "source": tool_res.get("source", "bocha") if isinstance(tool_res, dict) else "bocha"
                            }
                            search_tag = f"\n<search>{json.dumps(search_data, ensure_ascii=False)}</search>\n"
                            await put_event({
                                "type": "token",
                                "content": search_tag
                            })
                        else:
                            await put_event({
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
                        if is_single_search and name == "web_search":
                            search_data = {
                                "query": args.get("query", ""),
                                "results": [],
                                "source": "bocha"
                            }
                            search_tag = f"\n<search>{json.dumps(search_data, ensure_ascii=False)}</search>\n"
                            await put_event({
                                "type": "token",
                                "content": search_tag
                            })
                        else:
                            await put_event({
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
                    if not is_single_search:
                        await put_event({
                            "type": "plan_item",
                            "data": {"index": plan_index, "status": "completed"}
                        })
                    plan_index += 1

            # 4. Finish Event
            await put_event({"type": "done"})

        except Exception as e:
            await put_event({"type": "error", "content": f"Agent failed: {str(e)}"})

    async def _generate_plan(self, messages: List[Any], tools: List[Any]) -> Optional[Dict[str, Any]]:
        """Generates a step-by-step execution plan."""
        try:
            tool_descs = [{"name": t.name, "description": t.description} for t in tools]
            prompt = (
                "你是一个极其克制、甚至吝啬使用搜索引擎的任务规划助手。根据用户的问题和以下可用工具，制定一个执行计划。\n\n"
                f"可用工具：\n{json.dumps(tool_descs, ensure_ascii=False)}\n\n"
                "【规划核心原则——克制搜索，优先直接回复】\n"
                "1. 尽可能不要触发任何工具或网页搜索（web_search）。对于绝大多数常识性问题、技术概念解释、日常闲聊、创意写作、代码编写、翻译、逻辑推理，以及不需要实时新鲜资讯的问题，你必须通过已有知识储备直接回答。此时，你必须返回空列表：{\"items\": []}，绝对不要制定任何搜索计划。\n"
                "2. 只有在用户明确查询今天/最近发生的最新实时事件、近期新闻动态、实时天气、极其精确的时效性计算或当前确切系统时间等，且你的训练知识库中确实无法覆盖的领域时，才允许制定计划。\n"
                "3. 即使确实必须要执行搜索，也绝不允许安排超过一次网页搜索（web_search），严禁进行多次搜索或循环调用。\n"
                "4. 记住：直接回答永远是首选。只有在不搜索就完全无法回答的极少数情况下，才考虑使用单次搜索。\n\n"
                "请以 JSON 格式返回执行计划，格式如下：\n"
                '{"items": [{"id": 1, "description": "步骤描述", "tool_name": "工具名"}]}\n\n'
                '如果不需要任何工具或搜索，必须返回：{"items": []}\n'
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
