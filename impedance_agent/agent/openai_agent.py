# impedance_agent/agent/openai_agent.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base import BaseAgent
from ..core.env import env
from ..core.models import ImpedanceData
import logging
import json


class OpenAIAgent(BaseAgent):
    def __init__(self):
        """Initialize OpenAI agent"""
        super().__init__()
        config = env.get_provider_config("openai")
        if not config:
            raise ValueError("OpenAI provider not configured")

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model
        self.logger = logging.getLogger(__name__)

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert in electrochemical impedance analysis. Your goal is to "
            "run all available analyses sequentially.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST request ALL analyses in your FIRST response:\n"
            "- fit_linkk (LinKKResult)\n"
            "- fit_drt (DRTResult)\n"
            "- fit_ecm (FitResult) if model provided\n\n"
            "2. Do NOT wait for results before calling other tools.\n\n"
            "3. Analysis Flow:\n\n"
            "IF ECM FIT WAS PERFORMED:\n"
            "    A. Path Following Analysis (PRIMARY metric)\n"
            "    - Quantifies how well fit follows experimental arc shape\n"
            "    - Must be evaluated before any other metrics\n"
            "    - Rating Scale:\n"
            "        * Excellent: < 0.05 (5% path deviation)\n"
            "        * Acceptable: < 0.10 (10% path deviation)\n"
            "        * Poor: > 0.10 (INVALID MODEL)\n"
            "    - Critical Implications:\n"
            "        * Poor path following (> 0.10) means:\n"
            "            - Model structure is wrong\n"
            "            - Circuit elements are missing\n"
            "            - Cannot be fixed by parameter adjustment\n"
            "            - MUST modify model structure\n"
            "    - Quantitative assessment:\n"
            "        * ECM chi-square, AIC\n"
            "        * Parameter uncertainties\n"
            "        * Residual structure\n\n"
            "IF NO ECM FIT (Lin-KK and DRT only):\n"
            "    A. Data Quality Assessment (PRIMARY focus)\n"
            "    - Lin-KK validation metrics\n"
            "    - Residual analysis\n"
            "    - Error bounds verification\n"
            "    - Measurement quality evaluation\n\n"
            "    B. DRT Analysis\n"
            "    - Peak identification and characterization\n"
            "    - Time constant distribution assessment\n"
            "    - Physical process interpretation\n"
            "    - Model structure recommendations\n\n"
            "Secondary Analyses:\n\n"
            "1. Vector Difference Analysis (ECM fits only):\n"
            "- Only relevant if path following is acceptable\n"
            "- Rating Scale:\n"
            "    * Excellent: < 0.05 (5% average deviation)\n"
            "    * Acceptable: < 0.10 (10% average deviation)\n"
            "    * Poor: > 0.10\n\n"
            "2. Parameter Correlation Analysis (ECM fits only):\n"
            "- Special Cases (Expected Strong Correlations):\n"
            "    * CPE Parameters (Q, C and n):\n"
            "        - Strong Q-n correlation is EXPECTED and NOT overparameterization\n"
            "        - Check n value: n ≈ 1 means ideal capacitor, n < 1 means distributed\n"
            "        - Only consider fixing n if very close to 1 (> 0.90)\n"
            "    * Finite Warburg Parameters (W and tau):\n"
            "        - Strong W-tau correlation is EXPECTED and NOT overparameterization\n"
            "        - Represents physical relationship between diffusion coefficient and length\n"
            "- For other parameters:\n"
            "    * Strong correlation does NOT mean overparameterization unless uncertainties are huge\n"
            "    * Strong correlation might be related processes like solution and charge transfer resistance\n\n"
            "3. DRT Analysis:\n"
            "- Peak Analysis:\n"
            "    * Identify characteristic frequencies\n"
            "    * Each DRT peak and its likely physical origin\n"
            "- Use for model improvement:\n"
            "    * Correlations between DRT peaks and time constants\n"
            "    * Identify missing processes\n"
            "    * Guide structural changes\n\n"
            "4. Lin-KK Analysis:\n"
            "- Data quality validation\n"
            "- Residual structure assessment\n"
            "- Error bounds verification\n\n"
            "Integration Analysis:\n\n"
            "FOR ECM FITS:\n"
            "    IF Path Following is Poor (> 0.10):\n"
            "    - Model is INVALID\n"
            "    - Required Actions:\n"
            "        * No reason to check for overparameterization in non-CPE parameters\n"
            "        * Must change model structure\n"
            "        * Add elements based on DRT peaks\n"
            "        * Modify circuit topology\n"
            "        * Consider alternative models\n\n"
            "    IF Path Following is Acceptable:\n"
            "    - Proceed with:\n"
            "        * Vector difference analysis\n"
            "        * Parameter correlation assessment (remember CPE exception)\n"
            "        * Check for overparameterization in non-CPE parameters\n"
            "        * Parameter optimization considering uncertainties\n"
            "        * |r| > 0.9: Must simplify model\n"
            "        * 0.8 < |r| < 0.9: Investigate necessity\n"
            "        * Check physical meaning of correlations\n\n"
            "FOR NON-ECM ANALYSIS:\n"
            "    - Focus on data quality assessment\n"
            "    - Use DRT to recommend model structures\n"
            "    - Identify key time constants and processes\n"
            "    - Suggest appropriate circuit elements\n\n"
            "Final Report Requirements:\n\n"
            "FOR ECM FITS:\n"
            "1. Path Following Assessment:\n"
            "- State path difference value FIRST\n"
            "- Declare model validity/invalidity\n"
            "- Specify required structural changes if poor\n\n"
            "FOR ALL ANALYSES:\n"
            "2. Critical evaluation of:\n"
            "    * Data quality and measurement validity\n"
            "    * Model adequacy compared to DRT complexity\n"
            "    * Frequency-dependent behaviors and trends\n"
            "    * Possible measurement artifacts or system limitations\n"
            "    * Areas needing further investigation\n\n"
            "3. Specific Recommendations:\n"
            "For ECM analysis:\n"
            "    - Parameter optimization strategies\n"
            "    - Model structure improvements\n"
            "    - Physical interpretation guidance\n\n"
            "For non-ECM analysis:\n"
            "    - Data quality implications\n"
            "    - Suggested model structures\n"
            "    - Key time constants to consider\n"
            "    - Circuit element recommendations\n\n"
            "Remember:\n"
            "- First response: tool calls only\n"
            "- Analysis approach depends on available tools and results\n"
            "- CPE Q-n correlation is normal and expected\n"
            "- Strong correlation does NOT mean overparameterization unless uncertainties are huge\n"
            "- Use DRT to guide improvements\n"
            "- Consider physical meaning of parameters and processes\n"
            "- END analysis report with: \"NOTICE TO RESEARCHERS: LLMs hallucinate. All analyses and "
            "recommendations are intended as guidance to be evaluated alongside physical understanding "
            "and domain expertise.\""
        )

    def get_user_prompt(self, data: ImpedanceData, model_config: Optional[Dict]) -> str:
        prompt = (
            f"Analyze impedance data from {min(data.frequency):.1e} to {max(data.frequency):.1e} Hz.\n\n"
            f"Required analyses in order:\n\n"
        )

        if model_config:
            prompt += "1. fit_linkk (data validation)\n"
            prompt += "2. fit_drt (identify ALL time constants and processes)\n"
            prompt += "3. fit_ecm with this model:\n"
            prompt += f"```python\n{model_config['model_code']}\n```\n"
            prompt += "\nModel parameters:\n"
            for var in model_config["variables"]:
                prompt += (
                    f"- {var['name']}: initial={var['initialValue']}, "
                    f"bounds=[{var['lowerBound']}, {var['upperBound']}]\n"
                )
            prompt += "\nAfter fitting:\n"
            prompt += "- Compare number of DRT peaks vs ECM elements\n"
            prompt += "- Check if model accounts for all identified processes\n"
            prompt += "- Analyze residuals (real and imaginary) to assess fit quality\n"
            prompt += "- Suggest specific improvements if needed\n"
            prompt += "- Consider if additional circuit elements are needed\n"
            prompt += "- Evaluate if frequency ranges of processes match\n"
        else:
            prompt += "1. fit_linkk (data validation)\n"
            prompt += "2. fit_drt (identify ALL time constants and processes)\n"
            prompt += "\nAfter fitting:\n"
            prompt += "- Analyze residuals (real and imaginary) to assess fit quality\n"
            prompt += "- Provide overall assessment of data quality and reliability\n"

        prompt += "\n\nEnsure that residuals (both real and imaginary) are included in "
            "the analysis and evaluation of the fits."

        return prompt

    def _get_tools(self) -> List[Dict[str, Any]]:
        """OpenAI tool format with schema validation"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "fit_linkk",
                    "description": """
                    Validate impedance data quality using Lin-KK analysis.
                    Results will tell us if the data is reliable before further analysis.
                    Low residuals and good mu parameter indicate reliable data.
                    Check both real and imaginary components for consistency.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "c": {"type": "number", "default": 0.85},
                            "max_M": {"type": "integer", "default": 100},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fit_drt",
                    "description": """
                    Analyze distribution of relaxation times to identify ALL time constants.
                    This reveals the true number of processes in the system, which may be
                    more than what a simple ECM can model. Each peak might represent a distinct
                    physical process.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lambda_t": {"type": "number", "default": 1e-14},
                            "lambda_pg": {"type": "number", "default": 0.01},
                            "mode": {
                                "type": "string",
                                "enum": ["real", "imag"],
                                "default": "real",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fit_ecm",
                    "description": """
                    Fit equivalent circuit model to impedance data.
                    Compare results with DRT to assess if the model captures all processes.
                    Consider if additional elements are needed based on DRT peaks.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_code": {
                                "type": "string",
                                "description": "Complete Python function definition for the impedance model",
                            },
                            "variables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Parameter name",
                                        },
                                        "initialValue": {
                                            "type": "number",
                                            "description": "Initial guess for the parameter",
                                        },
                                        "lowerBound": {
                                            "type": "number",
                                            "description": "Lower bound for the parameter",
                                        },
                                        "upperBound": {
                                            "type": "number",
                                            "description": "Upper bound for the parameter",
                                        },
                                    },
                                    "required": [
                                        "name",
                                        "initialValue",
                                        "lowerBound",
                                        "upperBound",
                                    ],
                                },
                            },
                            "weighting": {
                                "type": "string",
                                "enum": ["unit", "proportional", "modulus"],
                                "default": "modulus",
                                "description": "Weighting scheme for the fit",
                            },
                        },
                        "required": ["model_code", "variables"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    def _format_message(self, message: Dict) -> Dict:
        """Format message for OpenAI API specifications"""
        try:
            # For assistant messages with tool calls
            if message.get("role") == "assistant":
                formatted = {"role": "assistant", "content": message.get("content", "")}
                if message.get("tool_calls"):
                    tool_calls = []
                    for call in message["tool_calls"]:
                        # Extract function name and arguments carefully
                        if hasattr(call, "function"):
                            func_name = call.function.name
                            arguments = call.function.arguments
                        else:
                            func_name = call["function"]["name"]
                            arguments = call["function"].get("arguments", {})

                        # Handle arguments properly
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, assume it's a raw string that needs to be preserved
                                arguments = arguments

                        # For ECM fits, ensure model_code is preserved as full function string
                        if func_name == "fit_ecm" and isinstance(arguments, dict):
                            # Preserve the full model code if it exists
                            if "model_code" in arguments and not arguments[
                                "model_code"
                            ].startswith("def "):
                                self.logger.warning(
                                    f"Model code appears truncated: {arguments['model_code']}"
                                )

                        # Construct the tool call
                        tool_call = {
                            "id": (
                                call.id if hasattr(call, "id") else f"call_{func_name}"
                            ),
                            "type": "function",
                            "function": {
                                "name": func_name,
                                # Ensure arguments are properly JSON serialized if they're a dict
                                "arguments": (
                                    json.dumps(arguments)
                                    if isinstance(arguments, dict)
                                    else arguments
                                ),
                            },
                        }
                        tool_calls.append(tool_call)
                    formatted["tool_calls"] = tool_calls
                return formatted

            # For tool responses
            if message.get("role") in ["tool", "function"]:
                return {
                    "role": "tool",
                    "tool_call_id": message.get(
                        "tool_call_id", f"call_{message['name']}"
                    ),
                    "name": message["name"],
                    "content": message["content"],
                }

            # For system/user messages, preserve them exactly as is
            return message.copy()

        except Exception as e:
            self.logger.error(f"Error formatting message: {str(e)}")
            self.logger.debug(f"Problematic message: {message}")
            raise

    def create_chat_completion(
        self, messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto"
    ) -> Any:
        """Make API call to OpenAI"""
        try:
            formatted_messages = []
            for msg in messages:
                try:
                    formatted = self._format_message(msg)
                    formatted_messages.append(formatted)
                except Exception as e:
                    self.logger.error(f"Failed to format message: {str(e)}")
                    self.logger.debug(f"Problematic message: {msg}")
                    raise

            self.logger.debug("Making OpenAI API call with formatted messages")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools if tools else self.tools,
                tool_choice=tool_choice,
            )
            return response

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            self.logger.exception("API call error details:")
            raise
