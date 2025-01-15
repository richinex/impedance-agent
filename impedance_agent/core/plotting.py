# src/core/plotting.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional, Union
from .models import AnalysisResult

# Constants for figure sizing
CM_TO_INCHES = 1 / 2.54  # conversion factor


class PlotManager:
    """Manages creation and export of publication-quality electrochemical analysis plots"""

    logger = logging.getLogger(__name__)

    @staticmethod
    def create_plots(
        result: AnalysisResult,
        output_dir: Union[str, Path],
        file_format: str = "png",
        dpi: int = 600,
        show: bool = False,
    ) -> None:
        """
        Generate and save publication-quality electrochemical analysis plots

        Parameters
        ----------
        result : AnalysisResult
            Analysis results containing impedance, DRT, and fitting data
        output_dir : Union[str, Path]
            Directory where plots will be saved
        file_format : str, optional
            Output file format (default: 'png')
        dpi : int, optional
            Resolution of saved plots (default: 600)
        show : bool, optional
            Whether to display plots (default: False)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        PlotManager.logger.info(f"Creating plots in {output_dir}")

        # Set publication-ready style
        plt.style.use("default")
        plt.rcParams.update(
            {
                "font.family": "Arial",
                "font.size": 7,  # Base font size
                "axes.labelsize": 8,  # Slightly larger for labels
                "axes.titlesize": 8,  # Same as labels
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "lines.linewidth": 0.75,  # Thinner lines
                "lines.markersize": 3,  # Smaller markers
                "axes.linewidth": 0.5,  # Thinner box edges
                "figure.dpi": 600,
                "savefig.dpi": 600,
                "figure.constrained_layout.use": False,  # Disable constrained layout
            }
        )

        # Create figure with subplots - single-column width
        fig_width = 16 * CM_TO_INCHES  # 8.5 cm for single-column width
        fig_height = 16 * CM_TO_INCHES  # Slightly taller for better spacing

        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.3)  # Increased spacing

        # Create all subplots
        axes = []
        for i in range(2):
            for j in range(2):
                axes.append(fig.add_subplot(gs[i, j]))

        # Generate individual plots
        PlotManager._plot_nyquist(result, axes[0])
        PlotManager._plot_drt(result, axes[1])
        PlotManager._plot_linkk(result, axes[2])
        PlotManager._plot_bode(result, axes[3])

        # Apply consistent styling to all plots
        for ax in axes:
            ax.grid(True, alpha=0.2, linestyle="--", color="gray")
            ax.set_axisbelow(True)
            if hasattr(ax, "legend_"):
                ax.legend(
                    loc="best",
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="none",
                    fancybox=True,
                )
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)  # Thinner spines for journal style

        # Save figure
        plot_path = output_dir / f"impedance_analysis.{file_format}"
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight", format=file_format)
        PlotManager.logger.info(f"Saved plots to {plot_path}")

        if show:
            plt.show()
        plt.close()

    @staticmethod
    def _plot_nyquist(result: AnalysisResult, ax) -> None:
        """Create publication-quality Nyquist plot"""
        if not hasattr(result, "linkk_fit") or result.linkk_fit is None:
            PlotManager.logger.warning("No Lin-KK results available for Nyquist plot")
            return

        if not hasattr(result.linkk_fit, "Z_fit"):
            PlotManager.logger.warning("No impedance data found in Lin-KK results")
            return

        Z = result.linkk_fit.Z_fit

        # Plot experimental data
        ax.plot(
            Z.real,
            -Z.imag,
            "o",
            color="#1f77b4",
            label="Experimental",
            markerfacecolor="none",
            markeredgewidth=0.75,
        )

        # Plot ECM fit if available with defensive check
        if (
            result.ecm_fit is not None
            and hasattr(result.ecm_fit, "Z_fit")
            and result.ecm_fit.Z_fit is not None
        ):
            z_fit = result.ecm_fit.Z_fit
            ax.plot(
                z_fit.real,
                -z_fit.imag,
                "-",
                color="#ff7f0e",
                label="ECM Fit",
                linewidth=0.75,
            )

        ax.set_xlabel(r"$\Re(Z)$ / $\Omega$", labelpad=2)
        ax.set_ylabel(r"$-\Im(Z)$ / $\Omega$", labelpad=2)
        ax.set(adjustable="datalim", aspect="equal")
        ax.legend(loc="best")
        ax.set_title("Nyquist Plot", pad=8)

    @staticmethod
    def _plot_drt(result: AnalysisResult, ax) -> None:
        """Create publication-quality Distribution of Relaxation Times (DRT) plot"""
        if not result.drt_fit:
            PlotManager.logger.warning("No DRT results available")
            return

        if not hasattr(result.drt_fit, "tau") or result.drt_fit.tau is None:
            PlotManager.logger.warning("No tau values found in DRT results")
            return

        if not hasattr(result.drt_fit, "gamma") or result.drt_fit.gamma is None:
            PlotManager.logger.warning("No gamma values found in DRT results")
            return

        freq = 1 / (2 * np.pi * result.drt_fit.tau)

        # Plot DRT distribution
        ax.plot(
            freq,
            result.drt_fit.gamma,
            "-",
            color="#1f77b4",
            linewidth=0.75,
            label="DRT",
        )

        # Mark peaks with defensive check
        if (
            hasattr(result.drt_fit, "peak_frequencies")
            and result.drt_fit.peak_frequencies is not None
            and hasattr(result.drt_fit, "peak_polarizations")
            and result.drt_fit.peak_polarizations is not None
        ):
            ax.scatter(
                result.drt_fit.peak_frequencies,
                result.drt_fit.peak_polarizations,
                color="#ff7f0e",
                marker="o",
                s=20,
                label="Peak Processes",
                zorder=5,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Frequency / Hz", labelpad=2)
        ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", labelpad=2)
        ax.legend(loc="best")
        ax.set_title("Distribution of Relaxation Times", pad=8)

    @staticmethod
    def _plot_linkk(result: AnalysisResult, ax) -> None:
        """Create publication-quality Lin-KK residuals plot"""
        if not result.linkk_fit:
            PlotManager.logger.warning("No Lin-KK results available")
            return

        if not result.drt_fit or not hasattr(result.drt_fit, "tau"):
            PlotManager.logger.warning(
                "No tau values available for frequency calculation"
            )
            return

        if not hasattr(result.linkk_fit, "residuals_real") or not hasattr(
            result.linkk_fit, "residuals_imag"
        ):
            PlotManager.logger.warning("No residuals found in Lin-KK results")
            return

        freq = 1 / (2 * np.pi * result.drt_fit.tau)

        # Plot residuals in percentage
        ax.plot(
            freq,
            result.linkk_fit.residuals_real * 100,
            "o",
            color="#1f77b4",
            label="Real",
            markerfacecolor="none",
            markeredgewidth=0.75,
        )
        ax.plot(
            freq,
            result.linkk_fit.residuals_imag * 100,
            "o",
            color="#ff7f0e",
            label="Imaginary",
            markerfacecolor="none",
            markeredgewidth=0.75,
        )

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency / Hz", labelpad=2)
        ax.set_ylabel("Relative Residuals / %", labelpad=2)
        ax.legend(loc="best")
        ax.set_title(
            f"Lin-KK Validation (M={getattr(result.linkk_fit, 'M', 'N/A')})", pad=8
        )
        ax.set_ylim(-5, 5)  # Set to -5% to 5% range

    @staticmethod
    def _plot_bode(result: AnalysisResult, ax) -> None:
        """Create publication-quality Bode plot"""
        if not result.linkk_fit:
            PlotManager.logger.warning("No Lin-KK results available for Bode plot")
            return

        if not hasattr(result.linkk_fit, "Z_fit"):
            PlotManager.logger.warning("No impedance data found in Lin-KK results")
            return

        if not result.drt_fit or not hasattr(result.drt_fit, "tau"):
            PlotManager.logger.warning(
                "No tau values available for frequency calculation"
            )
            return

        Z = result.linkk_fit.Z_fit
        freq = 1 / (2 * np.pi * result.drt_fit.tau)

        # Create twin axis for phase
        ax2 = ax.twinx()

        # Calculate magnitude and phase
        magnitude = np.abs(Z)
        phase = -np.rad2deg(np.arctan2(Z.imag, Z.real))

        # Plot experimental data
        (p1,) = ax.semilogx(
            freq,
            magnitude,
            "o",
            color="blue",
            label="|Z|",
            markerfacecolor="none",
            markeredgewidth=0.75,
        )
        (p2,) = ax2.semilogx(
            freq,
            phase,
            "o",
            color="orange",
            label="Phase",
            markerfacecolor="none",
            markeredgewidth=0.75,
        )

        # Add ECM fit if available with defensive check
        if (
            result.ecm_fit is not None
            and hasattr(result.ecm_fit, "Z_fit")
            and result.ecm_fit.Z_fit is not None
        ):
            z_fit = result.ecm_fit.Z_fit
            magnitude_fit = np.abs(z_fit)
            phase_fit = -np.rad2deg(np.arctan2(z_fit.imag, z_fit.real))

            ax.semilogx(
                freq, magnitude_fit, "-", color="red", label="|Z| fit", linewidth=0.75
            )
            ax2.semilogx(
                freq, phase_fit, "-", color="purple", label="Phase fit", linewidth=0.75
            )

        # Set labels and styling
        ax.set_xlabel("Frequency / Hz", labelpad=2)
        ax.set_ylabel(r"|Z| / $\Omega$", labelpad=2)
        ax2.set_ylabel(r"Phase / $\degree$", labelpad=2)

        # Color coordinate axes with adjusted positioning
        ax.yaxis.set_label_coords(-0.2, 0.5)
        ax2.yaxis.set_label_coords(1.2, 0.5)

        ax.yaxis.label.set_color(p1.get_color())
        ax2.yaxis.label.set_color(p2.get_color())
        ax.tick_params(axis="y", colors=p1.get_color(), which="both")
        ax2.tick_params(axis="y", colors=p2.get_color(), which="both")

        # Set limits and scaling
        ax2.set_ylim([-105, 25])
        magnitude_range = np.ptp(magnitude)
        ax.set_ylim(
            [
                np.min(magnitude) - 0.15 * magnitude_range,
                np.max(magnitude) + 0.15 * magnitude_range,
            ]
        )
        ax.set_xlim([freq.min() * 0.8, freq.max() * 1.2])

        # Add legend
        lines = [p1, p2]
        labels = [p1.get_label(), p2.get_label()]
        ax.legend(
            lines,
            labels,
            loc="best",
            frameon=True,
            framealpha=0.9,
            edgecolor="none",
            fancybox=True,
        )

        ax.set_title("Bode Plot", pad=8)
