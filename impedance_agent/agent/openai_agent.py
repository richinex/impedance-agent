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
        config = env.get_provider_config('openai')
        if not config:
            raise ValueError("OpenAI provider not configured")

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.model = config.model
        self.logger = logging.getLogger(__name__)

    @property
    def system_prompt(self) -> str:
        return """You are an expert in electrochemical impedance analysis. Your goal is to run all available analyses sequentially.

    CRITICAL INSTRUCTIONS:
    1. You MUST request ALL analyses in your FIRST response:
    - fit_linkk (LinKKResult)
    - fit_drt (DRTResult)
    - fit_ecm (FitResult) if model provided

    2. Do NOT wait for results before calling other tools.

    3. Analysis Flow:

    IF ECM FIT WAS PERFORMED:
        A. Path Following Analysis (PRIMARY metric)
        - Quantifies how well fit follows experimental arc shape
        - Must be evaluated before any other metrics
        - Rating Scale:
            * Excellent: < 0.05 (5% path deviation)
            * Acceptable: < 0.10 (10% path deviation)
            * Poor: > 0.10 (INVALID MODEL)
        - Critical Implications:
            * Poor path following (> 0.10) means:
                - Model structure is wrong
                - Circuit elements are missing
                - Cannot be fixed by parameter adjustment
                - MUST modify model structure
        - Quantitative assessment:
            * ECM chi-square, AIC
            * Parameter uncertainties
            * Residual structure

    IF NO ECM FIT (Lin-KK and DRT only):
        A. Data Quality Assessment (PRIMARY focus)
        - Lin-KK validation metrics
        - Residual analysis
        - Error bounds verification
        - Measurement quality evaluation

        B. DRT Analysis
        - Peak identification and characterization
        - Time constant distribution assessment
        - Physical process interpretation
        - Model structure recommendations

    Secondary Analyses:

    1. Vector Difference Analysis (ECM fits only):
    - Only relevant if path following is acceptable
    - Rating Scale:
        * Excellent: < 0.05 (5% average deviation)
        * Acceptable: < 0.10 (10% average deviation)
        * Poor: > 0.10

    2. Parameter Correlation Analysis (ECM fits only):
    - Special Cases (Expected Strong Correlations):
        * CPE Parameters (Q, C and n):
            - Strong Q-n correlation is EXPECTED and NOT overparameterization
            - Check n value: n ≈ 1 means ideal capacitor, n < 1 means distributed
            - Only consider fixing n if very close to 1 (> 0.90)
        * Finite Warburg Parameters (W and tau):
            - Strong W-tau correlation is EXPECTED and NOT overparameterization
            - Represents physical relationship between diffusion coefficient and length
    - For other parameters:
        * Strong correlation does NOT mean overparameterization unless the parameter uncertainties are huge
        * Strong correlation might be related processes like solution resistance and charge transfer resistance

    3. DRT Analysis:
    - Peak Analysis:
        * Identify characteristic frequencies
        * Each DRT peak and its likely physical origin
    - Use for model improvement:
        * Correlations between DRT peaks and time constants
        * Identify missing processes
        * Guide structural changes

    4. Lin-KK Analysis:
    - Data quality validation
    - Residual structure assessment
    - Error bounds verification

    Integration Analysis:

    FOR ECM FITS:
        IF Path Following is Poor (> 0.10):
        - Model is INVALID
        - Required Actions:
            * No reason to check for overparameterization in non-CPE parameters
            * Must change model structure
            * Add elements based on DRT peaks
            * Modify circuit topology
            * Consider alternative models

        IF Path Following is Acceptable:
        - Proceed with:
            * Vector difference analysis
            * Parameter correlation assessment (remember CPE exception)
            * Check for overparameterization in non-CPE parameters
            * Parameter optimization considering uncertainties
            * |r| > 0.9: Must simplify model
            * 0.8 < |r| < 0.9: Investigate necessity
            * Check physical meaning of correlations

    FOR NON-ECM ANALYSIS:
        - Focus on data quality assessment
        - Use DRT to recommend model structures
        - Identify key time constants and processes
        - Suggest appropriate circuit elements

    Final Report Requirements:

    FOR ECM FITS:
    1. Path Following Assessment:
    - State path difference value FIRST
    - Declare model validity/invalidity
    - Specify required structural changes if poor

    FOR ALL ANALYSES:
    2. Critical evaluation of:
        * Data quality and measurement validity
        * Model adequacy compared to DRT complexity
        * Frequency-dependent behaviors and trends
        * Possible measurement artifacts or system limitations
        * Areas needing further investigation

    3. Specific Recommendations:
    For ECM analysis:
        - Parameter optimization strategies
        - Model structure improvements
        - Physical interpretation guidance

    For non-ECM analysis:
        - Data quality implications
        - Suggested model structures
        - Key time constants to consider
        - Circuit element recommendations

    Remember:
    - First response: tool calls only
    - Analysis approach depends on available tools and results
    - CPE Q-n correlation is normal and expected
    - Strong correlation does NOT mean overparameterization unless uncertainties are huge
    - Use DRT to guide improvements
    - Consider physical meaning of parameters and processes
    - END analysis report with: "NOTICE TO RESEARCHERS: LLMs hallucinate. All analyses and recommendations are intended as guidance to be evaluated alongside physical understanding and domain expertise."""


    def get_user_prompt(self, data: ImpedanceData, model_config: Optional[Dict]) -> str:
        prompt = (f"Analyze impedance data from {min(data.frequency):.1e} to {max(data.frequency):.1e} Hz.\n\n"
                f"Required analyses in order:\n\n")

        if model_config:
            prompt += "1. fit_linkk (data validation)\n"
            prompt += "2. fit_drt (identify ALL time constants and processes)\n"
            prompt += "3. fit_ecm with this model:\n"
            prompt += f"```python\n{model_config['model_code']}\n```\n"
            prompt += "\nModel parameters:\n"
            for var in model_config['variables']:
                prompt += (f"- {var['name']}: initial={var['initialValue']}, "
                        f"bounds=[{var['lowerBound']}, {var['upperBound']}]\n")
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

        prompt += "\n\nEnsure that residuals (both real and imaginary) are included in the analysis and evaluation of the fits."

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
                            "max_M": {"type": "integer", "default": 100}
                        }
                    }
                }
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
                                "default": "real"
                            }
                        }
                    }
                }
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
                                "description": "Complete Python function definition for the impedance model"
                            },
                            "variables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Parameter name"
                                        },
                                        "initialValue": {
                                            "type": "number",
                                            "description": "Initial guess for the parameter"
                                        },
                                        "lowerBound": {
                                            "type": "number",
                                            "description": "Lower bound for the parameter"
                                        },
                                        "upperBound": {
                                            "type": "number",
                                            "description": "Upper bound for the parameter"
                                        }
                                    },
                                    "required": ["name", "initialValue", "lowerBound", "upperBound"]
                                }
                            },
                            "weighting": {
                                "type": "string",
                                "enum": ["unit", "proportional", "modulus"],
                                "default": "modulus",
                                "description": "Weighting scheme for the fit"
                            }
                        },
                        "required": ["model_code", "variables"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    def _format_message(self, message: Dict) -> Dict:
        """Format message for OpenAI API specifications"""
        try:
            # For assistant messages with tool calls
            if message.get('role') == 'assistant':
                formatted = {"role": "assistant", "content": message.get('content', '')}
                if message.get('tool_calls'):
                    tool_calls = []
                    for call in message['tool_calls']:
                        # Extract function name and arguments carefully
                        if hasattr(call, 'function'):
                            func_name = call.function.name
                            arguments = call.function.arguments
                        else:
                            func_name = call['function']['name']
                            arguments = call['function'].get('arguments', {})

                        # Handle arguments properly
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                # If JSON parsing fails, assume it's a raw string that needs to be preserved
                                arguments = arguments

                        # For ECM fits, ensure model_code is preserved as full function string
                        if func_name == 'fit_ecm' and isinstance(arguments, dict):
                            # Preserve the full model code if it exists
                            if 'model_code' in arguments and not arguments['model_code'].startswith('def '):
                                self.logger.warning(f"Model code appears truncated: {arguments['model_code']}")

                        # Construct the tool call
                        tool_call = {
                            "id": call.id if hasattr(call, 'id') else f"call_{func_name}",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                # Ensure arguments are properly JSON serialized if they're a dict
                                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments
                            }
                        }
                        tool_calls.append(tool_call)
                    formatted["tool_calls"] = tool_calls
                return formatted

            # For tool responses
            if message.get('role') in ['tool', 'function']:
                return {
                    "role": "tool",
                    "tool_call_id": message.get('tool_call_id', f"call_{message['name']}"),
                    "name": message['name'],
                    "content": message['content']
                }

            # For system/user messages, preserve them exactly as is
            return message.copy()

        except Exception as e:
            self.logger.error(f"Error formatting message: {str(e)}")
            self.logger.debug(f"Problematic message: {message}")
            raise

    def create_chat_completion(self, messages: List[Dict], tools: List[Dict] = None,
                             tool_choice: str = "auto") -> Any:
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
                tool_choice=tool_choice
            )
            return response

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            self.logger.exception("API call error details:")
            raise