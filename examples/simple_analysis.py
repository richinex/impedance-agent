# examples/simple_analysis.py
import numpy as np
from src.core.models import ImpedanceData
from src.agent.orchestrator import OrchestratorAgent


def main():
    # Create sample data
    freq = np.logspace(-2, 5, 50)
    # Simple Randles circuit simulation
    z_real = 1 + 2 / (1 + (2 * np.pi * freq * 1e-3) ** 2)
    z_imag = -2 * 2 * np.pi * freq * 1e-3 / (1 + (2 * np.pi * freq * 1e-3) ** 2)

    data = ImpedanceData(frequency=freq, real=z_real, imaginary=z_imag)

    # Model configuration
    model_config = {
        "model_code": """
        def impedance_model(p, f):
            w = 2 * np.pi * f
            Rs, Rct, Cdl = p
            Z = Rs + Rct / (1 + 1j * w * Cdl * Rct)
            return np.concatenate([Z.real, Z.imag])
        """,
        "variables": [
            {"name": "Rs", "initialValue": 1.0, "lowerBound": 0, "upperBound": 10},
            {"name": "Rct", "initialValue": 2.0, "lowerBound": 0, "upperBound": 10},
            {"name": "Cdl", "initialValue": 1e-3, "lowerBound": 0, "upperBound": 1},
        ],
    }

    # Run analysis
    agent = OrchestratorAgent()
    result = agent.analyze(data, model_config)

    # Print results
    print(result.summary)
    for rec in result.recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()
