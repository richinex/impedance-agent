# impedance_agent/agent/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from openai import OpenAI
from ..core.env import env
from ..core.models import ImpedanceData
from ..agent.tools.fitter_tools import FitterTools
import logging

class BaseAgent(ABC):
    """Abstract base class for provider-specific agents"""

    def __init__(self):
        """Initialize base agent components"""
        self.logger = logging.getLogger(__name__)
        self.fitter_tools = FitterTools()
        self.tools = self._get_tools()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get provider-specific system prompt"""
        pass

    @abstractmethod
    def get_user_prompt(self, data: ImpedanceData, model_config: Optional[Dict]) -> str:
        """
        Get provider-specific user prompt

        Args:
            data: Impedance data to analyze
            model_config: Optional ECM configuration

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Get provider-specific tool definitions

        Returns:
            List of tool definitions in provider-specific format
        """
        pass

    @abstractmethod
    def _format_message(self, message: Dict) -> Dict:
        """
        Format message according to provider requirements

        Args:
            message: Message to format

        Returns:
            Formatted message for the specific provider
        """
        pass

    @abstractmethod
    def create_chat_completion(self,
                             messages: List[Dict],
                             tools: List[Dict] = None,
                             tool_choice: str = "auto") -> Any:
        """
        Make API call to provider

        Args:
            messages: List of conversation messages
            tools: List of available tools
            tool_choice: How tools should be selected

        Returns:
            Provider-specific response object
        """
        pass

    def _handle_tool_call(self, tool_call: Any, data: ImpedanceData) -> Dict[str, Any]:
        """
        Handle tool calls - common across providers

        Args:
            tool_call: Tool call object from the LLM
            data: Impedance data to analyze

        Returns:
            Result of the tool execution

        Raises:
            ValueError: If tool is not recognized
        """
        tool_name = tool_call.function.name
        arguments = tool_call.function.arguments

        if isinstance(arguments, str):
            try:
                arguments = eval(arguments)
                self.logger.debug(f"Parsed arguments: {arguments}")
            except Exception as e:
                self.logger.error(f"Error parsing arguments: {str(e)}")
                self.logger.error(f"Raw arguments: {arguments}")
                raise

        try:
            if tool_name == "fit_ecm":
                self.logger.debug("Starting ECM fit")
                result = self.fitter_tools.run_ecm_fit(data, **arguments)
                self.logger.debug("ECM fit completed")
                return result
            elif tool_name == "fit_drt":
                self.logger.debug("Starting DRT analysis")
                result = self.fitter_tools.run_drt_fit(data, **arguments)
                self.logger.debug("DRT analysis completed")
                return result
            elif tool_name == "fit_linkk":
                self.logger.debug("Starting Lin-KK validation")
                result = self.fitter_tools.run_linkk_fit(data, **arguments)
                self.logger.debug("Lin-KK validation completed")
                return result
            else:
                error_msg = f"Unknown tool: {tool_name}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except Exception as e:
            self.logger.error(f"Error in tool execution: {str(e)}")
            self.logger.exception("Tool execution error details:")
            raise