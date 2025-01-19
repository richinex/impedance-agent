Technical Implementations
=========================

This document explains the methods and calculations used in impedance analysis.

Data Quality Assessment
-----------------------

Lin-KK Analysis
~~~~~~~~~~~~~~~

The Lin-KK validation uses the impedancepy package, implementing the method from Schönleber et al. [Schonleber2014]_. This implementation:

- Uses a Kramers-Kronig circuit model with ohmic resistor and RC elements
- Finds the best number of RC elements automatically
- Analyzes residuals to check data quality
- Confirms if measurements follow physical principles

Equivalent Circuit Model (ECM) Fitting
--------------------------------------

Parameter Estimation Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fitting process has these steps:

1. Parameter Transformation

   For bounded optimization:

   .. math::

       p_{\text{int}} = \log_{10}\left(\frac{p - lb}{1 - p/ub}\right)

       p_{\text{ext}} = \frac{lb + 10^p}{1 + 10^p/ub}

2. Objective Function

   Using weighted residuals:

   .. math::

       \text{WRSS} = \sum_{i=1}^N \frac{(Z_{\text{exp},i} - Z_{\text{model},i})^2}{\sigma_i^2}

3. Optimization

   Uses BFGS algorithm with weighted residuals and parameter bounds.

Weighting Schemes
~~~~~~~~~~~~~~~~~~

We offer three weighting options:

.. math::

    \sigma_i = \begin{cases}
    1 & \text{for unit weighting} \\
    |Z_{\text{exp},i}| & \text{for proportional weighting} \\
    \sqrt{(Re(Z_{\text{exp},i}))^2 + (Im(Z_{\text{exp},i}))^2} & \text{for modulus weighting}
    \end{cases}

Parameter Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~

Calculated using QR decomposition of the weighted Jacobian:

.. math::

    \sigma_j = \|R^{-1}_j\| \sqrt{\text{WRMS}}

where:
- R comes from QR decomposition of the weighted Jacobian
- WRMS is weighted root mean square error
- Jacobian elements: :math:`J_{ij} = \frac{\partial Z_i}{\partial p_j}`

Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~

Using the Hessian of the objective function:

.. math::

    C_{ij} = \frac{H^{-1}_{ij}}{\sqrt{H^{-1}_{ii}H^{-1}_{jj}}}

Understanding the values:
- :math:`|r| > 0.9`: Strong correlation
- :math:`0.7 < |r| < 0.9`: Medium correlation
- :math:`|r| < 0.7`: Weak correlation

Fit Quality Metrics
--------------------

Vector Difference Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Measures point-by-point agreement:

.. math::

    \text{VD} = \frac{1}{N}\sum_{i=1}^N \frac{|Z_{\text{fit},i} - Z_{\text{exp},i}|}{|Z_{\text{exp},i}|}

Quality guides:
- Excellent: < 0.05 (5% average deviation)
- Good: < 0.10 (10% average deviation)
- Poor: > 0.10

Path Following Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Checks if model follows data trajectory:

.. math::

    \text{PD} = \frac{1}{N-1}\sum_{i=1}^{N-1} \left|\frac{\Delta Z_{\text{fit},i}}{|\Delta Z_{\text{fit},i}|} - \frac{\Delta Z_{\text{exp},i}}{|\Delta Z_{\text{exp},i}|}\right|

Quality guides:
- Excellent: < 0.05 (5% path deviation)
- Good: < 0.10 (10% path deviation)
- Poor: > 0.10 (shows model structure issues)

Model Selection Metrics
------------------------

Akaike Information Criterion (AIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For different weighting schemes [Ingdal2019]_:

Unit weighting:

.. math::

    \text{AIC} = 2N\ln(2\pi) - 2N\ln(2N) + 2N + 2N\ln(\text{WRSS}) + 2k

Modulus/proportional weighting:

.. math::

    \text{AIC} = 2N\ln(2\pi) - 2N\ln(2N) + 2N - \sum\ln(w_i) + 2N\ln(\text{WRSS}) + 2(k+1)

Sigma weighting:

.. math::

    \text{AIC} = 2N\ln(2\pi) + \sum\ln(\sigma_i^2) + \text{WRSS} + 2k

where:
- N is number of data points
- k is number of model parameters
- WRSS is weighted residual sum of squares

Distribution of Relaxation Times (DRT)
--------------------------------------

We use Kulikovsky's method [Kulikovsky2020]_ for DRT analysis, which:

- Combines Tikhonov regularization with projected gradient method
- Handles the ill-posed nature of DRT calculations
- Ensures physically meaningful results (non-negative distribution)
- Provides fast calculations

The objective function is:

.. math::

    \text{Objective Function} = \|Z_{\text{exp}} - Z_{\text{fit}}\|^2 + \lambda \|L \gamma\|^2

where:
- λ is regularization parameter
- L is regularization operator
- γ is distribution of relaxation times

Implementation Details
----------------------

Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~

- BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm
- Bounded optimization through parameter transformation
- Automatic differentiation for gradients

Numerical Stability
~~~~~~~~~~~~~~~~~~~

- SVD for correlation matrix calculation
- QR decomposition for uncertainty estimation
- DRT regularization
- Parameter scaling

LLM Integration
---------------

The analysis workflow integrates these metrics with the LLM to:
- Evaluate model validity based on path following
- Guide model structure modifications
- Interpret parameter correlations
- Provide physically meaningful recommendations

The LLM system is structured to prioritize analysis based on:
1. Path following assessment (primary metric for ECM fits)
2. Data quality assessment (primary focus for non-ECM analysis)


References
----------

.. [Schonleber2014] Schönleber, M., et al. (2014). A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests. *Electrochimica Acta*, **131**, 20-27.

.. [Ingdal2019] Ingdal, M., Johnsen, R., & Harrington, D. A. (2019). The Akaike information criterion in weighted regression of immittance data. *Electrochimica Acta*, **317**, 648-653.

.. [Kulikovsky2020] Kulikovsky, A. (2020). PEM fuel cell distribution of relaxation times: A method for calculation and behavior of oxygen transport peak. *Physical Chemistry Chemical Physics*, **19**, 19131.

Notes
-----

- JAX helps with automatic differentiation and fast computation
- Error calculations assume normal distribution of residuals
- CPE and Warburg elements need special attention for correlations
- DRT needs careful selection of regularization parameter