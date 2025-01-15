# impedance_agent/agent/analysis.py
# impedance_agent/agent/analysis.py
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any
import asyncio
import logging
from ..core.models import (
    ImpedanceData,
    AnalysisResult,
    DRTResult,
    LinKKResult,
    FitResult,
)
from .deepseek_agent import DeepSeekAgent
from .openai_agent import OpenAIAgent


class ImpedanceAnalysisAgent:
    """Agent for analyzing impedance data using LLM providers"""

    def __init__(self, provider: str = "deepseek"):
        """Initialize with specified provider"""
        self.logger = logging.getLogger(__name__)
        self.provider = provider

        # Initialize provider-specific agent
        if provider == "deepseek":
            self.agent = DeepSeekAgent()
        elif provider == "openai":
            self.agent = OpenAIAgent()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.completed_tasks = set()

    async def execute_tool_calls_concurrently(
        self, tool_calls, data: ImpedanceData
    ) -> Dict[str, Any]:
        """Execute all tool calls concurrently"""
        self.logger.info(
            f"Starting concurrent execution of {len(tool_calls)} tool calls"
        )
        results = {}

        with ThreadPoolExecutor() as executor:
            futures = []
            for tool_call in tool_calls:
                if tool_call.function.name not in self.completed_tasks:
                    self.logger.info(f"Queueing task: {tool_call.function.name}")
                    futures.append(
                        (
                            tool_call,
                            executor.submit(
                                self.agent._handle_tool_call, tool_call, data
                            ),
                        )
                    )
                    self.completed_tasks.add(tool_call.function.name)

            for tool_call, future in futures:
                try:
                    result = future.result(timeout=300)
                    self.logger.info(f"Task completed: {tool_call.function.name}")
                    results[tool_call.function.name] = result
                except concurrent.futures.TimeoutError:
                    error_msg = f"Task {tool_call.function.name} timed out"
                    self.logger.error(error_msg)
                    results[tool_call.function.name] = f"Error: {error_msg}"
                except Exception as e:
                    self.logger.error(
                        f"Task failed {tool_call.function.name}: {str(e)}"
                    )
                    results[tool_call.function.name] = f"Error: {str(e)}"

        return results

    async def _execute_tool_call(self, tool_call, data: ImpedanceData) -> Any:
        """Execute a single tool call"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent._handle_tool_call, tool_call, data
            )
            self.logger.info(f"Task completed: {tool_call.function.name}")
            return result
        except Exception as e:
            self.logger.error(f"Task failed {tool_call.function.name}: {str(e)}")
            raise

    def analyze(
        self, data: ImpedanceData, model_config: Optional[Dict] = None
    ) -> AnalysisResult:
        """Analyze impedance data"""
        self.logger.info("Starting analysis")
        analysis_result = AnalysisResult()

        # Setup initial messages
        messages = [
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": self.agent.get_user_prompt(data, model_config)},
        ]

        try:
            if isinstance(self.agent, DeepSeekAgent):
                # Parallel execution for DeepSeek
                response = self.agent.create_chat_completion(
                    messages=messages, tools=self.agent.tools, tool_choice="auto"
                )
                message = response.choices[0].message
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": (
                            message.tool_calls
                            if hasattr(message, "tool_calls")
                            else None
                        ),
                    }
                )

                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Execute all tools concurrently
                    results = asyncio.run(
                        self.execute_tool_calls_concurrently(message.tool_calls, data)
                    )

                    # Add results to conversation
                    for tool_call in message.tool_calls:
                        result_content = str(
                            results.get(tool_call.function.name, "Error: No result")
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": result_content,
                            }
                        )

                    # Store results
                    for tool_name, result in results.items():
                        if tool_name == "fit_ecm" and isinstance(result, FitResult):
                            analysis_result.ecm_fit = result
                        elif tool_name == "fit_drt" and isinstance(result, DRTResult):
                            analysis_result.drt_fit = result
                        elif tool_name == "fit_linkk" and isinstance(
                            result, LinKKResult
                        ):
                            analysis_result.linkk_fit = result

            else:
                # Sequential execution for OpenAI
                required_tools = {"fit_linkk", "fit_drt"}
                if model_config:
                    required_tools.add("fit_ecm")
                completed_tools = set()

                while len(completed_tools) < len(required_tools):
                    # Get next tool call
                    response = self.agent.create_chat_completion(
                        messages=messages, tools=self.agent.tools, tool_choice="auto"
                    )
                    message = response.choices[0].message

                    if not hasattr(message, "tool_calls") or not message.tool_calls:
                        missing_tools = required_tools - completed_tools
                        error_msg = f"Model failed to request required analyses: {missing_tools}"
                        self.logger.warning(error_msg)
                        return AnalysisResult(summary=error_msg)

                    # Process tool call
                    tool_call = message.tool_calls[0]
                    tool_name = tool_call.function.name

                    if tool_name in completed_tools:
                        continue

                    # Add tool call to conversation
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )

                    # Execute tool
                    result = asyncio.run(self._execute_tool_call(tool_call, data))

                    # Store result
                    if tool_name == "fit_ecm" and isinstance(result, FitResult):
                        analysis_result.ecm_fit = result
                    elif tool_name == "fit_drt" and isinstance(result, DRTResult):
                        analysis_result.drt_fit = result
                    elif tool_name == "fit_linkk" and isinstance(result, LinKKResult):
                        analysis_result.linkk_fit = result

                    # Add tool result to conversation
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": str(result),
                        }
                    )

                    completed_tools.add(tool_name)

            # Get final interpretation
            summary_response = self.agent.create_chat_completion(
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": "Please provide your detailed interpretation of all results.",
                    }
                ],
                tools=self.agent.tools,
                tool_choice="none",
            )

            if summary_response.choices[0].message.content:
                analysis_result.summary = summary_response.choices[0].message.content

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg)
            analysis_result.summary = error_msg
            analysis_result.recommendations.append("Analysis failed to complete")

        return analysis_result
