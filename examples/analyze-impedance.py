# examples/analyze_impedance.py
import numpy as np
from src.core.models import ImpedanceData
from src.agent.analysis import ImpedanceAnalysisAgent


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
            w = 2 * jnp.pi * f
            Rs, Rct, Cdl = p
            Z = Rs + Rct / (1 + 1j * w * Cdl * Rct)
            return jnp.concatenate([Z.real, Z.imag])
        """,
        "variables": [
            {"name": "Rs", "initialValue": 1.0, "lowerBound": 0, "upperBound": 10},
            {"name": "Rct", "initialValue": 2.0, "lowerBound": 0, "upperBound": 10},
            {"name": "Cdl", "initialValue": 1e-3, "lowerBound": 0, "upperBound": 1},
        ],
    }

    # Run analysis
    agent = ImpedanceAnalysisAgent()
    result = agent.analyze(data, model_config)

    # Print results
    print(result.summary)
    if result.time_constant_analysis:
        print("\nTime Constant Analysis:")
        print(f"Matching score: {result.time_constant_analysis['matching_score']:.2f}")


if __name__ == "__main__":
    main()
