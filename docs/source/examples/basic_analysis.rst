Basic Analysis Example
======================

Here's a complete example of analyzing impedance data:

.. code-block:: python

   from impedance_agent import ImpedanceAnalysisAgent

   # Initialize the agent
   agent = ImpedanceAnalysisAgent()

   # Load and analyze data
   results = agent.analyze(
       data_file="examples/data/randles_circuit.txt",
       model_config="examples/models/randles.yaml"
   )

   # Print results
   print(results)