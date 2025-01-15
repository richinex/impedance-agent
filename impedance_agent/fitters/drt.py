# src/fitters/drt.py
import logging
import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import least_squares
from typing import Optional, Dict, Tuple
from ..core.models import ImpedanceData, DRTResult


class DRTFitter:
    def __init__(
        self,
        zexp_re,
        zexp_im,
        omg,
        lam_t0,
        lam_pg0,
        lower_bounds,
        upper_bounds,
        mode="real",
    ):
        self.logger = logging.getLogger(__name__)

        jax.config.update("jax_enable_x64", True)

        # Calculate scale factor
        self.rpol = zexp_re[-1] - zexp_re[0]
        if abs(self.rpol) < 1e-10:
            self.rpol = 1.0
            self.logger.warning("Rpol near zero, using default value of 1.0")

        # Store normalized data
        self.zexp_re = zexp_re
        self.zexp_im = zexp_im
        self.zexp_re_norm = zexp_re / self.rpol
        self.zexp_im_norm = zexp_im / self.rpol

        self.omg = omg
        self.mode = mode
        self.lam_t0 = lam_t0
        self.lam_pg0 = lam_pg0
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Updated iteration settings
        self.niter = 80
        self.flagiter = 0

        # Initialize arrays
        self.tau = 1.0 / self.omg
        self.ln_tau = jnp.log(self.tau)
        self.d_ln_tau = self._create_dmesh(self.ln_tau)
        self.d_tau = self._create_dmesh(self.tau)
        self.id_matrix = jnp.identity(self.omg.size, dtype=jnp.float64)
        self.a_matrix = jnp.zeros((self.omg.size, self.omg.size), dtype=jnp.float64)

        self.logger.info(f"Initialized DRT fitter with mode={mode}")
        self.logger.debug(f"Initial parameters: λT={lam_t0:.2e}, λPG={lam_pg0:.2e}")

        # Build matrices and get initial guess
        self._create_tikhonov_matrix()
        self.gfun_init = self.tikh_solver(
            self.lam_t0, self.a_mat_t_a, self.b_rhs, self.id_matrix
        )

    def _create_dmesh(self, grid):
        """Creates a mesh spacing array for the given grid."""
        dh = jnp.zeros(self.omg.size, dtype=jnp.float64)
        for j in range(1, self.omg.size - 1):
            dh = dh.at[j].set(0.5 * (grid[j + 1] - grid[j - 1]))

        # Handle endpoints
        dh = dh.at[0].set(0.5 * (grid[1] - grid[0]))
        dh = dh.at[-1].set(0.5 * (grid[-1] - grid[-2]))
        return dh

    def _am_row(self, omega_value):
        """Builds a single row of the system matrix for a given frequency."""
        prod = omega_value * self.tau
        if self.mode == "real":
            return self.d_ln_tau / (1.0 + prod**2)
        else:
            return prod * self.d_ln_tau / (1.0 + prod**2)

    def _create_tikhonov_matrix(self):
        """Builds the system matrix A, the Tikhonov matrix A^T A + lamT I, and RHS."""
        self.logger.debug("Building Tikhonov matrix")
        self.a_matrix = jax.vmap(self._am_row)(self.omg)
        self.a_matrix_t = self.a_matrix.transpose()
        self.a_mat_t_a = jnp.matmul(self.a_matrix_t, self.a_matrix)
        self.a_tikh = self.a_mat_t_a + self.lam_t0 * self.id_matrix

        if self.mode == "real":
            self.b_rhs = jnp.matmul(self.a_matrix_t, self.zexp_re_norm)
        else:
            self.b_rhs = jnp.matmul(self.a_matrix_t, self.zexp_im_norm)

    def tikh_solver(self, lam_t, a_mat_t_a, b_rhs, id_matrix):
        """Solve the Tikhonov-regularized equation."""
        lhs_matrix = a_mat_t_a + lam_t * id_matrix
        sol, residuals, rank, sv = jnp.linalg.lstsq(lhs_matrix, b_rhs, rcond=0)
        return sol

    def objective_function(self, g_vector, lhs_matrix, b_rhs):
        """Objective function for projected gradient."""
        residuals = jnp.matmul(lhs_matrix, g_vector) - b_rhs
        return jnp.sum(residuals**2)

    def find_lambda(self):
        """Scans over lambda_T range to find optimal regularization parameters."""
        self.logger.debug("Scanning lambda parameter range")
        kmax, lam_val = 25, 1e-25
        solnorm = jnp.zeros(kmax, dtype=jnp.float64)
        resid = jnp.zeros(kmax, dtype=jnp.float64)
        lam_t_arr = jnp.zeros(kmax, dtype=jnp.float64)
        lam_pg_arr = jnp.zeros(kmax, dtype=jnp.float64)

        for k in range(kmax):
            lam_val = lam_val * 10
            lam_t_arr = lam_t_arr.at[k].set(lam_val)
            gfun = self.tikh_solver(lam_val, self.a_mat_t_a, self.b_rhs, self.id_matrix)
            resid = resid.at[k].set(self.residual_norm(gfun))
            solnorm = solnorm.at[k].set(jnp.sqrt(jnp.sum(gfun**2)))
            lam_pg_arr = lam_pg_arr.at[k].set(1.0 / jnp.linalg.norm(self.a_tikh))

        return resid, solnorm, lam_t_arr, lam_pg_arr

    def residual_norm(self, g_vector):
        """Calculate the norm of the residual using (A^T A)g - A^T b."""
        lhs_prod = jnp.matmul(self.a_mat_t_a, g_vector)
        norm_res = jnp.sqrt(jnp.sum((lhs_prod - self.b_rhs) ** 2))
        return norm_res

    def encode(self, p, lb, ub):
        """Converts external parameters to internal parameters in log10 scale."""
        return jnp.log10((p - lb) / (1 - p / ub))

    def decode(self, p, lb, ub):
        """Converts internal parameters to external parameters."""
        return (lb + 10**p) / (1 + 10**p / ub)

    def jacobian_lsq(
        self, pvec, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
    ):
        """Compute the Jacobian of the Tikhonov residual function."""
        return jax.jacobian(self.tikh_residual)(
            jnp.array(pvec), lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
        )

    def pg_solver(self, lamvec, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix):
        """Projected gradient solver using Tikhonov solution as initial guess."""
        lam_t, lam_pg = lamvec
        g_init = self.tikh_solver(lam_t, a_mat_t_a, b_rhs, id_matrix)
        lhs_matrix_new = a_mat_t_a + lam_t * id_matrix

        pg = ProjectedGradient(
            fun=jax.jit(self.objective_function),
            projection=projection_non_negative,
            tol=1e-8,
            maxiter=self.niter * 1000,
            implicit_diff=True,
            jit=True,
        )

        self.logger.debug("Running projected gradient optimization")
        solution = pg.run(init_params=g_init, lhs_matrix=lhs_matrix_new, b_rhs=b_rhs)

        r_poly = jnp.sum(solution.params * d_ln_tau)
        return solution.params, r_poly, solution.state.iter_num

    def tikh_residual(
        self, lamvec_log, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
    ):
        lamvec_norm = self.decode(lamvec_log, lb, ub)
        g_vector, rpoly, iterations = self.pg_solver(
            lamvec_norm, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix
        )
        resid = jnp.matmul(self.a_tikh, g_vector) - self.b_rhs
        return resid

    def rpol_peaks(self, g_vector):
        """Find peaks in the DRT spectrum and calculate their parameters."""
        self.logger.debug("Analyzing DRT peaks")
        g_np = np.array(g_vector, copy=True)
        peaks, _ = find_peaks(g_np, prominence=0.01)
        widths = peak_widths(g_np, peaks, rel_height=1.0)

        integrals = jnp.zeros(peaks.size, dtype=jnp.float64)
        for n in range(peaks.size):
            low_bound = int(widths[2][n])
            up_bound = int(widths[3][n])
            integrals = integrals.at[n].set(
                jnp.sum(
                    g_vector[low_bound:up_bound] * self.d_ln_tau[low_bound:up_bound]
                )
            )

        peak_params = jnp.zeros((2, peaks.size), dtype=jnp.float64)
        peak_params = peak_params.at[0, :].set(
            jnp.flip(1.0 / (2.0 * jnp.pi * self.tau[peaks]))
        )
        peak_params = peak_params.at[1, :].set(jnp.flip(integrals))
        return peak_params

    # src/fitters/drt.py


import logging
import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_non_negative
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import least_squares
from typing import Optional, Dict, Tuple
from ..core.models import ImpedanceData, DRTResult


class DRTFitter:
    def __init__(
        self,
        zexp_re,
        zexp_im,
        omg,
        lam_t0,
        lam_pg0,
        lower_bounds,
        upper_bounds,
        mode="real",
    ):
        self.logger = logging.getLogger(__name__)

        jax.config.update("jax_enable_x64", True)

        # Calculate scale factor
        self.rpol = zexp_re[-1] - zexp_re[0]
        if abs(self.rpol) < 1e-10:
            self.rpol = 1.0
            self.logger.warning("Rpol near zero, using default value of 1.0")

        # Store normalized data
        self.zexp_re = zexp_re
        self.zexp_im = zexp_im
        self.zexp_re_norm = zexp_re / self.rpol
        self.zexp_im_norm = zexp_im / self.rpol

        self.omg = omg
        self.mode = mode
        self.lam_t0 = lam_t0
        self.lam_pg0 = lam_pg0
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        # Updated iteration settings
        self.niter = 80
        self.flagiter = 0

        # Initialize arrays
        self.tau = 1.0 / self.omg
        self.ln_tau = jnp.log(self.tau)
        self.d_ln_tau = self._create_dmesh(self.ln_tau)
        self.d_tau = self._create_dmesh(self.tau)
        self.id_matrix = jnp.identity(self.omg.size, dtype=jnp.float64)
        self.a_matrix = jnp.zeros((self.omg.size, self.omg.size), dtype=jnp.float64)

        self.logger.info(f"Initialized DRT fitter with mode={mode}")
        self.logger.debug(f"Initial parameters: λT={lam_t0:.2e}, λPG={lam_pg0:.2e}")

        # Build matrices and get initial guess
        self._create_tikhonov_matrix()
        self.gfun_init = self.tikh_solver(
            self.lam_t0, self.a_mat_t_a, self.b_rhs, self.id_matrix
        )

    def compute_normalized_residuals(
        self, Z_fit: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalized residuals using impedance modulus for normalization."""
        # Calculate impedance modulus for normalization
        Z_mod = np.sqrt(self.zexp_re**2 + self.zexp_im**2)

        # Compute normalized residuals using modulus
        residuals_real = (Z_fit.real - self.zexp_re) / Z_mod
        residuals_imag = (Z_fit.imag - self.zexp_im) / Z_mod

        return residuals_real, residuals_imag

    def _create_dmesh(self, grid):
        """Creates a mesh spacing array for the given grid."""
        dh = jnp.zeros(self.omg.size, dtype=jnp.float64)
        for j in range(1, self.omg.size - 1):
            dh = dh.at[j].set(0.5 * (grid[j + 1] - grid[j - 1]))

        # Handle endpoints
        dh = dh.at[0].set(0.5 * (grid[1] - grid[0]))
        dh = dh.at[-1].set(0.5 * (grid[-1] - grid[-2]))
        return dh

    def _am_row(self, omega_value):
        """Builds a single row of the system matrix for a given frequency."""
        prod = omega_value * self.tau
        if self.mode == "real":
            return self.d_ln_tau / (1.0 + prod**2)
        else:
            return prod * self.d_ln_tau / (1.0 + prod**2)

    def _create_tikhonov_matrix(self):
        """Builds the system matrix A, the Tikhonov matrix A^T A + lamT I, and RHS."""
        self.logger.debug("Building Tikhonov matrix")
        self.a_matrix = jax.vmap(self._am_row)(self.omg)
        self.a_matrix_t = self.a_matrix.transpose()
        self.a_mat_t_a = jnp.matmul(self.a_matrix_t, self.a_matrix)
        self.a_tikh = self.a_mat_t_a + self.lam_t0 * self.id_matrix

        if self.mode == "real":
            self.b_rhs = jnp.matmul(self.a_matrix_t, self.zexp_re_norm)
        else:
            self.b_rhs = jnp.matmul(self.a_matrix_t, self.zexp_im_norm)

    def tikh_solver(self, lam_t, a_mat_t_a, b_rhs, id_matrix):
        """Solve the Tikhonov-regularized equation."""
        lhs_matrix = a_mat_t_a + lam_t * id_matrix
        sol, residuals, rank, sv = jnp.linalg.lstsq(lhs_matrix, b_rhs, rcond=0)
        return sol

    def objective_function(self, g_vector, lhs_matrix, b_rhs):
        """Objective function for projected gradient."""
        residuals = jnp.matmul(lhs_matrix, g_vector) - b_rhs
        return jnp.sum(residuals**2)

    def find_lambda(self):
        """Scans over lambda_T range to find optimal regularization parameters."""
        self.logger.debug("Scanning lambda parameter range")
        kmax, lam_val = 25, 1e-25
        solnorm = jnp.zeros(kmax, dtype=jnp.float64)
        resid = jnp.zeros(kmax, dtype=jnp.float64)
        lam_t_arr = jnp.zeros(kmax, dtype=jnp.float64)
        lam_pg_arr = jnp.zeros(kmax, dtype=jnp.float64)

        for k in range(kmax):
            lam_val = lam_val * 10
            lam_t_arr = lam_t_arr.at[k].set(lam_val)
            gfun = self.tikh_solver(lam_val, self.a_mat_t_a, self.b_rhs, self.id_matrix)
            resid = resid.at[k].set(self.residual_norm(gfun))
            solnorm = solnorm.at[k].set(jnp.sqrt(jnp.sum(gfun**2)))
            lam_pg_arr = lam_pg_arr.at[k].set(1.0 / jnp.linalg.norm(self.a_tikh))

        return resid, solnorm, lam_t_arr, lam_pg_arr

    def residual_norm(self, g_vector):
        """Calculate the norm of the residual using (A^T A)g - A^T b."""
        lhs_prod = jnp.matmul(self.a_mat_t_a, g_vector)
        norm_res = jnp.sqrt(jnp.sum((lhs_prod - self.b_rhs) ** 2))
        return norm_res

    def encode(self, p, lb, ub):
        """Converts external parameters to internal parameters in log10 scale."""
        return jnp.log10((p - lb) / (1 - p / ub))

    def decode(self, p, lb, ub):
        """Converts internal parameters to external parameters."""
        return (lb + 10**p) / (1 + 10**p / ub)

    def jacobian_lsq(
        self, pvec, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
    ):
        """Compute the Jacobian of the Tikhonov residual function."""
        return jax.jacobian(self.tikh_residual)(
            jnp.array(pvec), lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
        )

    def pg_solver(self, lamvec, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix):
        """Projected gradient solver using Tikhonov solution as initial guess."""
        lam_t, lam_pg = lamvec
        g_init = self.tikh_solver(lam_t, a_mat_t_a, b_rhs, id_matrix)
        lhs_matrix_new = a_mat_t_a + lam_t * id_matrix

        pg = ProjectedGradient(
            fun=jax.jit(self.objective_function),
            projection=projection_non_negative,
            tol=1e-8,
            maxiter=self.niter * 1000,
            implicit_diff=True,
            jit=True,
        )

        self.logger.debug("Running projected gradient optimization")
        solution = pg.run(init_params=g_init, lhs_matrix=lhs_matrix_new, b_rhs=b_rhs)

        r_poly = jnp.sum(solution.params * d_ln_tau)
        return solution.params, r_poly, solution.state.iter_num

    def tikh_residual(
        self, lamvec_log, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix, lb, ub
    ):
        lamvec_norm = self.decode(lamvec_log, lb, ub)
        g_vector, rpoly, iterations = self.pg_solver(
            lamvec_norm, lhs_matrix, a_mat_t_a, b_rhs, d_ln_tau, id_matrix
        )
        resid = jnp.matmul(self.a_tikh, g_vector) - self.b_rhs
        return resid

    def rpol_peaks(self, g_vector):
        """Find peaks in the DRT spectrum and calculate their parameters."""
        self.logger.debug("Analyzing DRT peaks")
        g_np = np.array(g_vector, copy=True)
        peaks, _ = find_peaks(g_np, prominence=0.01)
        widths = peak_widths(g_np, peaks, rel_height=1.0)

        integrals = jnp.zeros(peaks.size, dtype=jnp.float64)
        for n in range(peaks.size):
            low_bound = int(widths[2][n])
            up_bound = int(widths[3][n])
            integrals = integrals.at[n].set(
                jnp.sum(
                    g_vector[low_bound:up_bound] * self.d_ln_tau[low_bound:up_bound]
                )
            )

        peak_params = jnp.zeros((2, peaks.size), dtype=jnp.float64)
        peak_params = peak_params.at[0, :].set(
            jnp.flip(1.0 / (2.0 * jnp.pi * self.tau[peaks]))
        )
        peak_params = peak_params.at[1, :].set(jnp.flip(integrals))
        return peak_params

    def z_model_imre(self, g_vector):
        """Calculate the model impedance from the DRT solution."""
        z_mod = jnp.zeros(self.omg.size, dtype=jnp.float64)
        for i in range(self.omg.size):
            prod = self.omg[i] * self.tau
            if self.mode == "real":
                integrand = g_vector / (1.0 + prod**2)
            else:
                integrand = prod * g_vector / (1.0 + prod**2)
            z_mod = z_mod.at[i].set(jnp.sum(self.d_ln_tau * integrand))

        return jnp.flip(self.rpol * z_mod)

    def tikh_residual_norm(self, g_vector, lam_t, a_mat_t_a, b_rhs, id_matrix):
        """Compute norm of Tikhonov residual and LHS norm."""
        lhs_matrix = a_mat_t_a + lam_t * id_matrix
        lhs_prod = jnp.matmul(lhs_matrix, g_vector)
        sum_res = jnp.sqrt(jnp.sum((lhs_prod - b_rhs) ** 2))
        sum_lhs = jnp.sqrt(jnp.sum(lhs_prod**2))
        return sum_res, sum_lhs

    def fit(self) -> Optional[DRTResult]:
        """Perform the complete DRT fitting process."""
        try:
            self.logger.info("Starting DRT analysis")

            # Find optimal lambda parameters
            resid, solnorm, arr_lam_t, arr_lam_pg = self.find_lambda()
            self.logger.debug("Completed lambda parameter search")

            # Prepare for least squares
            lamvec_init = jnp.array([self.lam_t0, self.lam_pg0], dtype=jnp.float64)
            lamvec_init_log = self.encode(
                lamvec_init, self.lower_bounds, self.upper_bounds
            )

            # Perform optimization
            res_parm = least_squares(
                jax.jit(self.tikh_residual),
                lamvec_init_log,
                method="lm",
                jac=self.jacobian_lsq,
                args=(
                    self.a_tikh,
                    self.a_mat_t_a,
                    self.b_rhs,
                    self.d_ln_tau,
                    self.id_matrix,
                    self.lower_bounds,
                    self.upper_bounds,
                ),
            )

            final_lamvec = self.decode(res_parm.x, self.lower_bounds, self.upper_bounds)
            self.logger.info(
                f"Final parameters: λT={final_lamvec[0]:.2e}, λPG={final_lamvec[1]:.2e}"
            )

            # Get final DRT distribution
            gfun_final, rpoly, n_iters = self.pg_solver(
                final_lamvec,
                self.a_tikh,
                self.a_mat_t_a,
                self.b_rhs,
                self.d_ln_tau,
                self.id_matrix,
            )

            self.logger.info(f"Completed projected gradient in {n_iters} iterations")
            self.logger.info(f"Rpol = {self.rpol:.6g}, Final rpoly = {rpoly:.6g}")

            # Calculate residuals
            res_init, lhs_init = self.tikh_residual_norm(
                self.gfun_init, self.lam_t0, self.a_mat_t_a, self.b_rhs, self.id_matrix
            )
            res_fin, lhs_fin = self.tikh_residual_norm(
                gfun_final, final_lamvec[0], self.a_mat_t_a, self.b_rhs, self.id_matrix
            )

            self.logger.info(
                f"Residuals: initial = {res_init:.6e}, final = {res_fin:.6e}"
            )

            if res_parm.status > 0:
                self.logger.info(
                    f"Optimization successful: {res_parm.njev} Jacobian evaluations"
                )

            if self.flagiter == 1:
                self.logger.warning(
                    "Maximum iteration limit reached in projected gradient"
                )

            # Find peaks
            peak_params = self.rpol_peaks(gfun_final)
            if peak_params.size > 0:
                peak_freqs = [float(f) for f in peak_params[0]]
                peak_pols = [float(p) for p in peak_params[1]]
                self.logger.info("Identified DRT peaks:")
                for f, p in zip(peak_freqs, peak_pols):
                    self.logger.info(f"  f = {f:.2f} Hz, polarization = {p:.5f}")
            else:
                self.logger.warning("No peaks detected in DRT spectrum")
                peak_freqs, peak_pols = [], []

            # Convert JAX arrays to numpy arrays
            tau = np.array(self.tau)
            gamma = np.array(gfun_final)

            # Calculate model impedance
            Z_fit = self.z_model_imre(gfun_final)
            Z_fit = np.array(Z_fit)  # Convert to numpy array

            # Calculate normalized residuals
            residuals_real, residuals_imag = self.compute_normalized_residuals(Z_fit)

            # Validate results
            if np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
                raise ValueError("Invalid values detected in final DRT solution")

            # Create result object with fitted impedance and residuals
            result = DRTResult(
                tau=tau,
                gamma=gamma,
                peak_frequencies=peak_freqs,
                peak_polarizations=peak_pols,
                regularization_param=float(final_lamvec[0]),
                residual=float(rpoly),
                Z_fit=Z_fit,
                residuals_real=residuals_real,
                residuals_imag=residuals_imag,
            )

            self.logger.debug(f"Created DRTResult object with {len(peak_freqs)} peaks")
            return result

        except Exception as e:
            self.logger.error("DRT fitting failed", exc_info=True)
            return None

    def tikh_residual_norm(self, g_vector, lam_t, a_mat_t_a, b_rhs, id_matrix):
        """Compute norm of Tikhonov residual and LHS norm."""
        lhs_matrix = a_mat_t_a + lam_t * id_matrix
        lhs_prod = jnp.matmul(lhs_matrix, g_vector)
        sum_res = jnp.sqrt(jnp.sum((lhs_prod - b_rhs) ** 2))
        sum_lhs = jnp.sqrt(jnp.sum(lhs_prod**2))
        return sum_res, sum_lhs

    def fit(self) -> Optional[DRTResult]:
        """Perform the complete DRT fitting process."""
        try:
            self.logger.info("Starting DRT analysis")

            # Find optimal lambda parameters
            resid, solnorm, arr_lam_t, arr_lam_pg = self.find_lambda()
            self.logger.debug("Completed lambda parameter search")

            # Prepare for least squares
            lamvec_init = jnp.array([self.lam_t0, self.lam_pg0], dtype=jnp.float64)
            lamvec_init_log = self.encode(
                lamvec_init, self.lower_bounds, self.upper_bounds
            )

            # Perform optimization
            res_parm = least_squares(
                jax.jit(self.tikh_residual),
                lamvec_init_log,
                method="lm",
                jac=self.jacobian_lsq,
                args=(
                    self.a_tikh,
                    self.a_mat_t_a,
                    self.b_rhs,
                    self.d_ln_tau,
                    self.id_matrix,
                    self.lower_bounds,
                    self.upper_bounds,
                ),
            )

            final_lamvec = self.decode(res_parm.x, self.lower_bounds, self.upper_bounds)
            self.logger.info(
                f"Final parameters: λT={final_lamvec[0]:.2e}, λPG={final_lamvec[1]:.2e}"
            )

            # Get final DRT distribution
            gfun_final, rpoly, n_iters = self.pg_solver(
                final_lamvec,
                self.a_tikh,
                self.a_mat_t_a,
                self.b_rhs,
                self.d_ln_tau,
                self.id_matrix,
            )

            self.logger.info(f"Completed projected gradient in {n_iters} iterations")
            self.logger.info(f"Rpol = {self.rpol:.6g}, Final rpoly = {rpoly:.6g}")

            # Calculate residuals
            res_init, lhs_init = self.tikh_residual_norm(
                self.gfun_init, self.lam_t0, self.a_mat_t_a, self.b_rhs, self.id_matrix
            )
            res_fin, lhs_fin = self.tikh_residual_norm(
                gfun_final, final_lamvec[0], self.a_mat_t_a, self.b_rhs, self.id_matrix
            )

            self.logger.info(
                f"Residuals: initial = {res_init:.6e}, final = {res_fin:.6e}"
            )

            if res_parm.status > 0:
                self.logger.info(
                    f"Optimization successful: {res_parm.njev} Jacobian evaluations"
                )

            if self.flagiter == 1:
                self.logger.warning(
                    "Maximum iteration limit reached in projected gradient"
                )

            # Find peaks
            peak_params = self.rpol_peaks(gfun_final)
            if peak_params.size > 0:
                peak_freqs = [float(f) for f in peak_params[0]]
                peak_pols = [float(p) for p in peak_params[1]]
                self.logger.info("Identified DRT peaks:")
                for f, p in zip(peak_freqs, peak_pols):
                    self.logger.info(f"  f = {f:.2f} Hz, polarization = {p:.5f}")
            else:
                self.logger.warning("No peaks detected in DRT spectrum")
                peak_freqs, peak_pols = [], []

            # Convert JAX arrays to numpy arrays
            tau = np.array(self.tau)
            gamma = np.array(gfun_final)

            # Calculate model impedance
            Z_fit = self.z_model_imre(gfun_final)
            Z_fit = np.array(Z_fit)  # Convert to numpy array

            # Calculate normalized residuals
            residuals_real, residuals_imag = self.compute_normalized_residuals(Z_fit)

            # Validate results
            if np.any(np.isnan(gamma)) or np.any(np.isinf(gamma)):
                raise ValueError("Invalid values detected in final DRT solution")

            # Create result object with fitted impedance and residuals
            result = DRTResult(
                tau=tau,
                gamma=gamma,
                peak_frequencies=peak_freqs,
                peak_polarizations=peak_pols,
                regularization_param=float(final_lamvec[0]),
                residual=float(rpoly),
                Z_fit=Z_fit,
                residuals_real=residuals_real,
                residuals_imag=residuals_imag,
            )

            self.logger.debug(f"Created DRTResult object with {len(peak_freqs)} peaks")
            return result

        except Exception as e:
            self.logger.error("DRT fitting failed", exc_info=True)
            return None
