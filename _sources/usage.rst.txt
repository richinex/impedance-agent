Usage
=====

Command Line Interface
======================

Basic Analysis
--------------

The CLI provides several options for analyzing impedance data:

.. code-block:: bash

    # Basic analysis with default settings
    impedance-agent analyze data.txt

    # Analysis with custom ECM model
    impedance-agent analyze data.txt --ecm model.yaml

    # Analysis with specific LLM provider
    impedance-agent analyze data.txt --provider deepseek

    # Export results and generate plots
    impedance-agent analyze data.txt --output-path results/analysis.json --plot

CLI Options
-----------

.. code-block:: text

    Arguments:
      data_path              Path to impedance data file

    Options:
      --provider TEXT        LLM provider (deepseek/openai) [default: deepseek]
      --ecm TEXT            Path to the equivalent circuit model(ECM) configuration file
      --output-path TEXT    Path for output files
      --output-format TEXT  Output format (json/csv/excel) [default: json]
      --plot-format TEXT    Plot format (png/pdf/svg) [default: png]
      --plot               Generate plots [default: True]
      --show-plots         Display plots in window [default: False]
      --log-level TEXT     Logging level
      --debug              Enable debug mode
      --workers INTEGER    Number of worker processes

Python API
==========

Basic Usage
-----------

.. code-block:: python

    from impedance_agent import ImpedanceAnalysisAgent

    # Initialize the agent with specific provider
    agent = ImpedanceAnalysisAgent(provider="deepseek")

    # Analyze data with built-in model
    results = agent.analyze("data.txt", model="randles")

Complete Example
----------------

Here's a complete example demonstrating impedance analysis with synthetic data:

.. code-block:: python

    import numpy as np
    from impedance_agent.core.models import ImpedanceData
    from impedance_agent.agent.analysis import ImpedanceAnalysisAgent

    # Create sample data
    freq = np.logspace(-2, 5, 50)
    z_real = 1 + 2 / (1 + (2 * np.pi * freq * 1e-3) ** 2)
    z_imag = -2 * 2 * np.pi * freq * 1e-3 / (1 + (2 * np.pi * freq * 1e-3) ** 2)

    data = ImpedanceData(frequency=freq, real=z_real, imaginary=z_imag)

    # Define ECM configuration
    ecm_config = {
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
    agent = ImpedanceAnalysisAgent(provider="deepseek")
    result = agent.analyze(data, ecm_config)

    # Print results
    print(result.summary)
    if result.time_constant_analysis:
        print("\nTime Constant Analysis:")
        print(f"Matching score: {result.time_constant_analysis['matching_score']:.2f}")