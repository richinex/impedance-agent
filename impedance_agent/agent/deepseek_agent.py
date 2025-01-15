# impedance_agent/agent/deepseek_agent.py
# impedance_agent/agent/deepseek_agent.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base import BaseAgent
from ..core.env import env
from ..core.models import ImpedanceData


class DeepSeekAgent(BaseAgent):
    def __init__(self):
        """Initialize DeepSeek agent"""
        super().__init__()
        config = env.get_provider_config("deepseek")
        if not config:
            raise ValueError("DeepSeek provider not configured")

        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.model = config.model

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert in electrochemical impedance analysis. Your goal is to "
            "run all available analyses concurrently.\n\n"
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
            f"Analyze impedance data from {min(data.frequency):.1e} to "
            f"{max(data.frequency):.1e} Hz.\n\n"
            f"REQUIRED: Call these tools simultaneously in your first response:\n"
        )

        if model_config:
            prompt += "\n1. fit_linkk - for data validation"
            prompt += "\n2. fit_drt - for time constant distribution"
            prompt += "\n3. fit_ecm - using this model configuration:\n"
            prompt += f"```python\n{model_config['model_code']}\n```\n"
            prompt += "Variables:\n"
            for var in model_config["variables"]:
                prompt += (
                    f"- {var['name']}: initial={var['initialValue']}, "
                    f"bounds=[{var['lowerBound']}, {var['upperBound']}]\n"
                )
        else:
            prompt += "\n1. fit_linkk - for data validation"
            prompt += "\n2. fit_drt - for time constant distribution"

        prompt += (
            "\n\nEnsure that residuals (both real and imaginary) are included in "
            "the analysis and evaluation of the fits."
        )

        return prompt

    def _get_tools(self) -> List[Dict[str, Any]]:
        """DeepSeek tool format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "fit_linkk",
                    "description": "Perform Lin-KK validation analysis",
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
                    "name": "fit_ecm",
                    "description": "Fit equivalent circuit model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_code": {"type": "string"},
                            "variables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "initialValue": {"type": "number"},
                                        "lowerBound": {"type": "number"},
                                        "upperBound": {"type": "number"},
                                    },
                                },
                            },
                            "weighting": {
                                "type": "string",
                                "enum": ["unit", "proportional", "modulus"],
                                "default": "modulus",
                            },
                        },
                        "required": ["model_code", "variables"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fit_drt",
                    "description": "Perform DRT analysis",
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
        ]

    def _format_message(self, message: Dict) -> Dict:
        """Format message for DeepSeek"""
        # For tool results
        if message.get("role") in ["tool", "function"]:
            return {
                "role": "tool",
                "tool_call_id": message.get("tool_call_id", f"call_{message['name']}"),
                "name": message["name"],
                "content": message["content"],
            }

        # For assistant messages
        if message.get("role") == "assistant":
            formatted = {"role": "assistant", "content": message.get("content", "")}
            if message.get("tool_calls"):
                formatted["tool_calls"] = message["tool_calls"]
            return formatted

        # For all other messages
        return message.copy()

    def create_chat_completion(
        self, messages: List[Dict], tools: List[Dict] = None, tool_choice: str = "auto"
    ) -> Any:
        """Make API call to DeepSeek"""
        try:
            formatted_messages = []
            tool_calls_ids = set()

            # Track and validate message sequence
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    tool_calls_ids.update(
                        call.id if hasattr(call, "id") else f"call_{call['name']}"
                        for call in msg["tool_calls"]
                    )
                formatted_messages.append(self._format_message(msg))

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools if tools else self.tools,
                tool_choice=tool_choice,
            )
            return response
        except Exception as e:
            self.logger.error(f"DeepSeek API call failed: {str(e)}")
            self.logger.exception("API call error details:")
            raise
