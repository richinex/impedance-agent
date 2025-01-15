# src/fitters/ecm.py
import jax
import jaxopt
import jax.numpy as jnp
import numpy as np
import logging
from typing import Optional, Tuple
from ..core.models import ImpedanceData, FitResult, FitQualityMetrics


class ECMFitter:
    """
    Equivalent Circuit Model (ECM) fitting for electrochemical impedance data.

    This class implements least squares optimization with bounded parameters for fitting
    equivalent circuit models to impedance data. The fitting process uses weighted residuals
    and supports different weighting schemes.

    The fundamental equation for the weighted sum of squared residuals is:

    .. math::
        WRSS = \\sum_{i=1}^N \\frac{(Z_{\\mathrm{exp},i} - Z_{\\mathrm{model},i})^2}{\\sigma_i^2}

    Supported weighting schemes:

    .. math::
        \\sigma_i = \\begin{cases}
        1 & \\text{for unit weighting} \\\\
        |Z_{\\mathrm{exp},i}| & \\text{for proportional weighting} \\\\
        \\sqrt{(\\mathrm{Re}(Z_{\\mathrm{exp},i}))^2 + (\\mathrm{Im}(Z_{\\mathrm{exp},i}))^2} & \\text{for modulus weighting}
        \\end{cases}

    Parameters
    ----------
    model_func : callable
        Function that takes parameters and frequencies and returns impedance
    p0 : array_like
        Initial parameter values
    freq : array_like
        Frequency values
    impedance_data : ImpedanceData
        Object containing experimental impedance data
    lb : array_like
        Lower bounds for parameters
    ub : array_like
        Upper bounds for parameters
    param_info : list
        List of dictionaries containing parameter information
    weighting : str or array_like, optional
        Weighting scheme ('unit', 'proportional', 'modulus') or custom weights (default: 'modulus')
    """

    def __init__(
        self,
        model_func,
        p0,
        freq,
        impedance_data: ImpedanceData,
        lb,
        ub,
        param_info,
        weighting="modulus",
    ):
        """
        Initialize ECM fitter with model and data.

        Parameters
        ----------
        model_func : callable
            Function that takes parameters and frequencies and returns impedance
        p0 : array_like
            Initial parameter values
        freq : array_like
            Frequency values
        impedance_data : ImpedanceData
            Object containing experimental impedance data
        lb : array_like
            Lower bounds for parameters
        ub : array_like
            Upper bounds for parameters
        param_info : list
            List of dictionaries containing parameter information
        weighting : str or array_like, optional
            Weighting scheme ('unit', 'proportional', 'modulus') or custom weights (default: 'modulus')
        """
        self.logger = logging.getLogger(__name__)
        jax.config.update("jax_enable_x64", True)

        # Validate bounds
        if any(lb > ub):
            raise ValueError("Lower bounds must be less than upper bounds")

        # Store data and setup
        self.impedance_data = impedance_data
        self.freq = freq
        self.data = jnp.array(impedance_data.real) + 1j * jnp.array(
            impedance_data.imaginary
        )
        self.model = jax.jit(model_func)
        self.p0 = p0
        self.lb = lb
        self.ub = ub
        self.num_params = len(p0)
        self.num_freq = len(freq)
        self.dof = 2 * self.num_freq - self.num_params
        self.param_info = param_info

        # Set up weighting
        self._setup_weighting(weighting)

    def compute_normalized_residuals(
        self, Z_fit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized residuals using impedance modulus for normalization.

        .. math::
            r_{real} = \\frac{Z_{fit,real} - Z_{exp,real}}{\\sqrt{Z_{exp,real}^2 + Z_{exp,imag}^2}}

            r_{imag} = \\frac{Z_{fit,imag} - Z_{exp,imag}}{\\sqrt{Z_{exp,real}^2 + Z_{exp,imag}^2}}

        Parameters
        ----------
        Z_fit : np.ndarray
            Complex array containing the fitted impedance values

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - residuals_real: Normalized residuals of the real component
            - residuals_imag: Normalized residuals of the imaginary component
        """
        # Calculate impedance modulus |Z| for normalization
        Z_mod = np.sqrt(self.impedance_data.real**2 + self.impedance_data.imaginary**2)

        # Compute normalized residuals using modulus
        residuals_real = (Z_fit.real - self.impedance_data.real) / Z_mod
        residuals_imag = (Z_fit.imag - self.impedance_data.imaginary) / Z_mod

        return residuals_real, residuals_imag

    def _setup_weighting(self, weighting):
        """
        Configure weighting scheme for the fitting process.

        Parameters
        ----------
        weighting : str or array_like
            Weighting scheme ('unit', 'proportional', 'modulus') or custom weights

        .. math::
            \\sigma_i = \\begin{cases}
                1 & \\text{for unit weighting} \\\\
                |Z_{exp,i}| & \\text{for proportional weighting} \\\\
                \\sqrt{(Re(Z_{exp,i}))^2 + (Im(Z_{exp,i}))^2} & \\text{for modulus weighting}
            \\end{cases}
        """
        if isinstance(weighting, (jnp.ndarray, np.ndarray)):
            self.logger.debug("Using custom sigma weighting array")
            self.weighting_name = "sigma"
            weighting = jnp.array(weighting)
            assert (
                self.data.shape == weighting.shape
            ), "Shape mismatch between data and weight array"
            self.zerr_Re = weighting
            self.zerr_Im = weighting
        elif isinstance(weighting, str):
            assert weighting.lower() in [
                "unit",
                "proportional",
                "modulus",
            ], f"Invalid weighting type: {weighting}"
            self.weighting_name = weighting.lower()
            self.logger.info(f"Using {self.weighting_name} weighting")
            if weighting.lower() == "unit":
                self.zerr_Re = jnp.ones(self.num_freq)
                self.zerr_Im = jnp.ones(self.num_freq)
            elif weighting.lower() == "proportional":
                self.zerr_Re = jnp.abs(self.data.real)
                self.zerr_Im = jnp.abs(self.data.imag)
            else:  # modulus weighting
                self.zerr_Re = jnp.abs(self.data)
                self.zerr_Im = jnp.abs(self.data)

    def encode(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Convert external parameters to internal parameters using log-transform.

        .. math::
            p_{int} = \\log_{10}\\left(\\frac{p - lb}{1 - p/ub}\\right)

        Parameters
        ----------
        p : jnp.ndarray
            External parameters

        Returns
        -------
        jnp.ndarray
            Internal parameters
        """
        return jnp.log10((p - self.lb) / (1 - p / self.ub))

    def decode(self, p: jnp.ndarray) -> jnp.ndarray:
        """
        Convert internal parameters to external parameters.

        .. math::
            p_{ext} = \\frac{lb + 10^p}{1 + 10^p/ub}

        Parameters
        ----------
        p : jnp.ndarray
            Internal parameters

        Returns
        -------
        jnp.ndarray
            External parameters
        """
        return (self.lb + 10**p) / (1 + 10**p / self.ub)

    def obj_fun(self, p_log: jnp.ndarray) -> float:
        """
        Calculate weighted objective function.

        .. math::
            WRSS = \\sum_{i=1}^N \\frac{(Z_{exp,i} - Z_{model,i})^2}{\\sigma_i^2}

        Parameters
        ----------
        p_log : jnp.ndarray
            Parameters in log space

        Returns
        -------
        float
            Weighted residual sum of squares
        """
        p_norm = self.decode(p_log)
        z_concat = jnp.concatenate([self.data.real, self.data.imag])
        sigma = jnp.concatenate([self.zerr_Re, self.zerr_Im])
        z_model = self.model(p_norm, self.freq)
        wrss = jnp.sum((1 / sigma**2) * (z_concat - z_model) ** 2)
        return wrss

    def compute_aic(self, wrss: float) -> float:
        """
        Compute Akaike Information Criterion (AIC) for model selection.

        For unit weighting:

        .. math::
            \\mathrm{AIC} = 2N\\ln(2\\pi) - 2N\\ln(2N) + 2N + 2N\\ln(\\mathrm{WRSS}) + 2k

        For modulus/proportional weighting:

        .. math::
            \\mathrm{AIC} = 2N\\ln(2\\pi) - 2N\\ln(2N) + 2N - \\sum\\ln(w_i) + 2N\\ln(\\mathrm{WRSS}) + 2(k+1)

        For sigma weighting:

        .. math::
            \\mathrm{AIC} = 2N\\ln(2\\pi) + \\sum\\ln(\\sigma_i^2) + \\mathrm{WRSS} + 2k

        Parameters
        ----------
        wrss : float
            Weighted residual sum of squares

        Returns
        -------
        float
            Computed AIC value
    """
        wt_re = 1 / self.zerr_Re**2
        wt_im = 1 / self.zerr_Im**2

        if self.weighting_name == "sigma":
            m2lnL = (
                (2 * self.num_freq) * jnp.log(2 * jnp.pi)
                + jnp.sum(jnp.log(self.zerr_Re**2))
                + jnp.sum(jnp.log(self.zerr_Im**2))
                + wrss
            )
            return m2lnL + 2 * self.num_params
        elif self.weighting_name == "unit":
            m2lnL = (
                2 * self.num_freq * jnp.log(2 * jnp.pi)
                - 2 * self.num_freq * jnp.log(2 * self.num_freq)
                + 2 * self.num_freq
                + 2 * self.num_freq * jnp.log(wrss)
            )
            return m2lnL + 2 * self.num_params
        else:
            m2lnL = (
                2 * self.num_freq * jnp.log(2 * jnp.pi)
                - 2 * self.num_freq * jnp.log(2 * self.num_freq)
                + 2 * self.num_freq
                - jnp.sum(jnp.log(wt_re))
                - jnp.sum(jnp.log(wt_im))
                + 2 * self.num_freq * jnp.log(wrss)
            )
            return m2lnL + 2 * (self.num_params + 1)

    def _calculate_uncertainties(self, popt: jnp.ndarray, wrms: float) -> jnp.ndarray:
        """
        Calculate parameter uncertainties using QR decomposition of the Jacobian.

        .. math::
            \\sigma_j = \\|R^{-1}_j\\| \\sqrt{WRMS}

        Where:
        - R is the upper triangular matrix from QR decomposition
        - WRMS is the weighted root mean square error

        Parameters
        ----------
        popt : jnp.ndarray
            Optimal parameters
        wrms : float
            Weighted root mean square error

        Returns
        -------
        jnp.ndarray
            Array of parameter uncertainties
        """
        grads = jax.jacfwd(self.model)(popt, self.freq)
        grads_re = grads[: self.num_freq]
        grads_im = grads[self.num_freq :]

        rtwre = jnp.diag(1 / self.zerr_Re)
        rtwim = jnp.diag(1 / self.zerr_Im)
        vre = rtwre @ grads_re
        vim = rtwim @ grads_im

        Q1, R1 = jnp.linalg.qr(jnp.concatenate([vre, vim], axis=0))
        invR1 = jnp.linalg.inv(R1)
        return jnp.linalg.norm(invR1, axis=1) * jnp.sqrt(wrms)

    def _calculate_correlation_matrix(self, popt: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate correlation matrix using Hessian of objective function.

        .. math::
            C_{ij} = \\frac{H^{-1}_{ij}}{\\sqrt{H^{-1}_{ii}H^{-1}_{jj}}}

        Where H is the Hessian matrix at the optimal parameters.

        Parameters
        ----------
        popt : jnp.ndarray
            Optimal parameters

        Returns
        -------
        jnp.ndarray
            Correlation matrix
        """
        # Get Hessian at optimal parameters
        hessian = jax.hessian(self.obj_fun)(self.encode(popt))

        # Use SVD for numerical stability
        U, s, Vt = jnp.linalg.svd(hessian, full_matrices=False)

        # Filter small singular values
        rcond = jnp.finfo(s.dtype).eps * max(hessian.shape)
        cutoff = rcond * s[0]
        s_inv = jnp.where(s > cutoff, 1 / s, 0)

        # Compute covariance matrix as inverse of Hessian
        cov = (Vt.T * s_inv) @ Vt

        # Calculate correlation matrix
        std = jnp.sqrt(jnp.diag(cov))
        corr = cov / (std[:, None] @ std[None, :])

        return corr

    def calculate_fitted_impedance(self, parameters: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate fitted impedance values from parameters.

        Parameters
        ----------
        parameters : jnp.ndarray
            Model parameters

        Returns
        -------
        jnp.ndarray
            Complex array of fitted impedance values
        """
        z_model = self.model(parameters, self.freq)
        return z_model[: self.num_freq] + 1j * z_model[self.num_freq :]

    def compute_fit_quality_metrics(self, Z_fit: np.ndarray) -> FitQualityMetrics:
        """
        Compute fit quality using vector difference and path deviation metrics.

        The vector difference analysis quantifies the point-by-point agreement:

        .. math::
            \\mathrm{VD} = \\frac{1}{N}\\sum_{i=1}^N \\frac{|Z_{\\mathrm{fit},i} - Z_{\\mathrm{exp},i}|}{|Z_{\\mathrm{exp},i}|}

        The path deviation analysis quantifies trajectory agreement:

        .. math::
            \\mathrm{PD} = \\frac{1}{N-1}\\sum_{i=1}^{N-1} \\left|\\frac{\\Delta Z_{\\mathrm{fit},i}}{|\\Delta Z_{\\mathrm{fit},i}|} -
            \\frac{\\Delta Z_{\\mathrm{exp},i}}{|\\Delta Z_{\\mathrm{exp},i}|}\\right|

        Parameters
        ----------
        Z_fit : np.ndarray
            Complex fitted impedance array

        Returns
        -------
        FitQualityMetrics
            Computed quality metrics including vector difference, path deviation,
            and overall quality assessment
        """
        Z_exp = self.impedance_data.real + 1j * self.impedance_data.imaginary

        # 1. Vector Difference Analysis
        vector_diff = np.mean(np.abs(Z_fit - Z_exp) / np.abs(Z_exp))

        # Assign vector quality
        if vector_diff < 0.05:  # 5% average deviation
            vector_quality = "excellent"
        elif vector_diff < 0.10:  # 10% average deviation
            vector_quality = "acceptable"
        else:
            vector_quality = "poor"

        # 2. Path Deviation Analysis
        dZ_exp = np.diff(Z_exp)
        dZ_fit = np.diff(Z_fit)

        # Normalize vectors to unit length and compare directions
        path_diff = np.mean(np.abs(dZ_fit / np.abs(dZ_fit) - dZ_exp / np.abs(dZ_exp)))

        # Assign path quality
        if path_diff < 0.05:  # 5% average path deviation
            path_quality = "excellent"
        elif path_diff < 0.10:  # 10% average path deviation
            path_quality = "acceptable"
        else:
            path_quality = "poor"

        # Overall quality assessment
        if vector_quality == "excellent" and path_quality == "excellent":
            overall_quality = "excellent"
        elif vector_quality == "poor" or path_quality == "poor":
            overall_quality = "poor"
        else:
            overall_quality = "acceptable"

        return FitQualityMetrics(
            vector_difference=float(vector_diff),
            vector_quality=vector_quality,
            path_deviation=float(path_diff),
            path_quality=path_quality,
            overall_quality=overall_quality,
        )

    def fit(self) -> Optional[FitResult]:
        """
        Perform impedance fitting on the data.

        The fitting process involves:
        1. Parameter optimization using BFGS algorithm
        2. Uncertainty calculation via QR decomposition
        3. Computation of fit quality metrics
        4. Calculation of correlation matrix
        5. Model selection metrics (AIC)

        Returns
        -------
        Optional[FitResult]
            Complete fitting results including:
            - Optimized parameters and uncertainties
            - Correlation matrix
            - Goodness-of-fit metrics (χ², AIC, WRMS)
            - Fitted impedance values
            - Fit quality assessment
            Returns None if fitting fails
        """
        try:
            self.logger.info("Starting ECM fitting")

            # Convert initial parameters to log scale
            p_log = self.encode(self.p0)
            self.logger.debug(f"Initial parameters (log scale): {p_log}")

            # Optimize using BFGS
            solver = jaxopt.ScipyMinimize(method="BFGS", fun=jax.jit(self.obj_fun))
            sol = solver.run(p_log)

            # Get optimized parameters
            popt_log = sol.params
            popt = self.decode(popt_log)
            wrss = sol.state.fun_val
            wrms = wrss / self.dof

            self.logger.info(f"Optimization complete: WRMS = {wrms:.6e}")

            param_info_str = "\n".join(
                [f"{p['name']}: {val:.6e}" for p, val in zip(self.param_info, popt)]
            )
            self.logger.debug(f"Optimal parameters:\n{param_info_str}")

            # Calculate uncertainties
            perr = self._calculate_uncertainties(popt, wrms)

            # Calculate fitted impedance values
            Z_fit = self.calculate_fitted_impedance(popt)
            Z_fit = np.array(Z_fit)

            # Calculate normalized residuals
            residuals_real, residuals_imag = self.compute_normalized_residuals(Z_fit)

            # Compute AIC
            aic = self.compute_aic(wrss)

            # Compute fit quality metrics
            fit_quality_metrics = self.compute_fit_quality_metrics(Z_fit)

            # Calculate correlation matrix
            correlation_matrix = np.array(self._calculate_correlation_matrix(popt))

            # Create result object with all metrics
            result = FitResult(
                parameters=popt.tolist(),
                errors=perr.tolist(),
                param_info=self.param_info,
                correlation_matrix=correlation_matrix,
                chi_square=float(wrss),
                aic=float(aic),
                wrms=float(wrms),
                dof=self.dof,
                Z_fit=Z_fit,
                fit_quality=fit_quality_metrics,
            )

            self.logger.info(f"Fit metrics: χ² = {wrss:.6e}, AIC = {aic:.6e}")
            self.logger.debug(
                "Parameter uncertainties:\n"
                + "\n".join(
                    [f"{p['name']}: {err:.6e}" for p, err in zip(self.param_info, perr)]
                )
            )

            return result

        except Exception:
            self.logger.error("ECM fitting failed", exc_info=True)
            return None