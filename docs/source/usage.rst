Usage
=====

Command Line Interface
======================

Basic Analysis
==============

.. code-block:: bash

   impedance-agent analyze data.txt --ecm model.yaml

Python API
==========

Basic Usage
===========

.. code-block:: python

   from impedance_agent import ImpedanceAnalysisAgent

   # Initialize the agent
   agent = ImpedanceAnalysisAgent()

   # Analyze data
   results = agent.analyze("data.txt", model="randles")