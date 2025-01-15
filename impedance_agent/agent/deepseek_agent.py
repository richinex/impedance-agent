# impedance_agent/agent/deepseek_agent.py
# impedance_agent/agent/deepseek_agent.py
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base import BaseAgent
from ..core.env import env
from ..core.models import ImpedanceData
import logging

class DeepSeekAgent(BaseAgent):
    def __init__(self):
        """Initialize DeepSeek agent"""
        super().__init__()
        config = env.get_provider_config('deepseek')
        if not config:
            raise ValueError("DeepSeek provider not configured")

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.model = config.model

    @property
    def system_prompt(self) -> str:
        return """You are an expert in electrochemical impedance analysis. Your goal is to run all available analyses concurrently for maximum efficiency.

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
        prompt = (f"Analyze impedance data from {min(data.frequency):.1e} to "
                f"{max(data.frequency):.1e} Hz.\n\n"
                f"REQUIRED: Call these tools simultaneously in your first response:\n")

        if model_config:
            prompt += "\n1. fit_linkk - for data validation"
            prompt += "\n2. fit_drt - for time constant distribution"
            prompt += "\n3. fit_ecm - using this model configuration:\n"
            prompt += f"```python\n{model_config['model_code']}\n```\n"
            prompt += "Variables:\n"
            for var in model_config['variables']:
                prompt += (f"- {var['name']}: initial={var['initialValue']}, "
                        f"bounds=[{var['lowerBound']}, {var['upperBound']}]\n")
        else:
            prompt += "\n1. fit_linkk - for data validation"
            prompt += "\n2. fit_drt - for time constant distribution"

        prompt += "\n\nEnsure that residuals (both real and imaginary) are included in the analysis and evaluation of the fits."

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
                            "max_M": {"type": "integer", "default": 100}
                        }
                    }
                }
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
                                        "upperBound": {"type": "number"}
                                    }
                                }
                            },
                            "weighting": {
                                "type": "string",
                                "enum": ["unit", "proportional", "modulus"],
                                "default": "modulus"
                            }
                        },
                        "required": ["model_code", "variables"]
                    }
                }
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
                                "default": "real"
                            }
                        }
                    }
                }
            }
        ]

    def _format_message(self, message: Dict) -> Dict:
        """Format message for DeepSeek"""
        # For tool results
        if message.get('role') in ['tool', 'function']:
            return {
                "role": "tool",
                "tool_call_id": message.get('tool_call_id', f"call_{message['name']}"),
                "name": message['name'],
                "content": message['content']
            }

        # For assistant messages
        if message.get('role') == 'assistant':
            formatted = {"role": "assistant", "content": message.get('content', '')}
            if message.get('tool_calls'):
                formatted["tool_calls"] = message['tool_calls']
            return formatted

        # For all other messages
        return message.copy()

    def create_chat_completion(self, messages: List[Dict], tools: List[Dict] = None,
                             tool_choice: str = "auto") -> Any:
        """Make API call to DeepSeek"""
        try:
            formatted_messages = []
            tool_calls_ids = set()

            # Track and validate message sequence
            for msg in messages:
                if msg.get('role') == 'assistant' and msg.get('tool_calls'):
                    tool_calls_ids.update(
                        call.id if hasattr(call, 'id') else f"call_{call['name']}"
                        for call in msg['tool_calls']
                    )
                formatted_messages.append(self._format_message(msg))

            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                tools=tools if tools else self.tools,
                tool_choice=tool_choice
            )
            return response
        except Exception as e:
            self.logger.error(f"DeepSeek API call failed: {str(e)}")
            self.logger.exception("API call error details:")
            raise