# impedance_agent/cli/main.py
# Standard library imports
import asyncio
import asyncio.exceptions
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Third-party imports
import typer

# Matplotlib imports (with config)
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Local imports
from ..agent.analysis import ImpedanceAnalysisAgent
from ..core.config import Config
from ..core.env import env
from ..core.exceptions import ExportError
from ..core.exporters import ResultExporter
from ..core.loaders import ImpedanceLoader
from ..core.logging import setup_logging
from ..core.plotting import PlotManager


app = typer.Typer()


async def run_async_tasks(
    result, output_path, output_format, plot, plot_dir, plot_format, show_plots
):
    """Run export and plotting tasks concurrently"""
    try:
        # Set up and execute tasks
        if output_path:
            # Execute both export tasks concurrently
            await asyncio.gather(
                ResultExporter.export_async(result, output_path, output_format),
                ResultExporter.export_async(
                    result, Path(output_path).parent / "analysis_summary.md", "md"
                ),
            )

        if plot and plot_dir:
            try:
                await asyncio.to_thread(
                    PlotManager.create_plots,
                    result=result,
                    output_dir=plot_dir,
                    file_format=plot_format,
                    show=show_plots,
                    dpi=300,
                )
            except ExportError as e:
                logging.error(f"Plot export failed: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during plotting: {e}")

    except ExportError as e:
        logging.error(f"Export failed: {e}")
    except Exception as e:
        logging.error(f"Error in async tasks: {str(e)}")


@app.command()
def analyze(
    data_path: str = typer.Argument(..., help="Path to impedance data file"),
    provider: str = typer.Option("deepseek", help="LLM provider (deepseek/openai)"),
    ecm: Optional[str] = typer.Option(
        None, help="Path to the equivalent circuit model(ECM) configuration file"
    ),
    output_path: Optional[str] = typer.Option(None, help="Path for output files"),
    output_format: str = typer.Option("json", help="Output format (json/csv/excel)"),
    plot_format: str = typer.Option("png", help="Plot format (png/pdf/svg)"),
    plot: bool = typer.Option(True, help="Generate plots"),
    show_plots: bool = typer.Option(False, help="Display plots in window"),
    log_level: str = typer.Option(env.log_level, help="Logging level"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
    workers: int = typer.Option(None, help="Number of worker processes"),
):
    """Analyze impedance data using ECM and/or DRT analysis with parallel processing"""
    loop = None
    try:
        # Setup logging
        level = "DEBUG" if debug else log_level
        setup_logging(level)
        logger = logging.getLogger(__name__)

        # Validate provider
        available_providers = env.get_available_providers()
        if not available_providers:
            logger.error(
                "No LLM providers are properly configured. Please check your .env file."
            )
            raise typer.Exit(1)

        if provider not in available_providers:
            logger.error(
                f"Provider '{provider}' not available. Available: {', '.join(available_providers)}"
            )
            raise typer.Exit(1)

        # Early validation of plot settings
        if plot and not output_path:
            logger.warning(
                "Plot generation requires an output path. Creating default 'results' directory."
            )
            output_path = "results/analysis.json"

        # Create output directories if needed
        plot_dir = None
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            if plot:
                plot_dir = output_dir / "plots"
                plot_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Plot files will be saved to: {plot_dir}")

        if debug:
            logger.debug(f"Loading file: {data_path}")
            logger.debug(f"Using provider: {provider}")
            if ecm:
                logger.debug(f"Using equivalent circuit model config: {ecm}")

        # Determine optimal number of workers if not specified
        if workers is None:
            workers = min(os.cpu_count(), 4)  # Use up to 4 cores by default

        # Load data and config concurrently
        async def load_data_and_config():
            with ThreadPoolExecutor(max_workers=2) as pool:
                loop = asyncio.get_running_loop()
                data_future = loop.run_in_executor(
                    pool, ImpedanceLoader.load, data_path
                )

                if ecm:
                    config_future = loop.run_in_executor(pool, Config.load_model, ecm)
                    data, ecm_cfg = await asyncio.gather(data_future, config_future)
                else:
                    data = await data_future
                    ecm_cfg = None

                return data, ecm_cfg

        # Set up event loop with policy
        if os.name == "posix":
            policy = asyncio.get_event_loop_policy()
            policy.set_event_loop(policy.new_event_loop())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Load data
        data, ecm_cfg = loop.run_until_complete(load_data_and_config())

        # Run analysis with specified provider
        logger.info(f"Starting impedance analysis using {provider}")
        agent = ImpedanceAnalysisAgent(provider=provider)
        result = agent.analyze(data, ecm_cfg)

        # Run export and plotting concurrently
        if output_path or (plot and plot_dir):
            loop.run_until_complete(
                run_async_tasks(
                    result,
                    output_path,
                    output_format,
                    plot,
                    plot_dir,
                    plot_format,
                    show_plots,
                )
            )

        # Handle plot display if requested
        if show_plots and not matplotlib.is_interactive():
            plt.ion()
            plt.show(block=False)
            plt.pause(0.1)  # Give time for display

        # Print summary to console
        typer.echo("\nAnalysis Summary:")
        typer.echo(result.summary)

    except asyncio.CancelledError:
        logger.error("Async operation was cancelled")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise typer.Exit(1)
    finally:
        # Cleanup
        if loop and loop.is_running():
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception as e:
                logger.error(f"Error during task cleanup: {str(e)}")

        # Close plots and loop
        plt.close("all")
        if loop and not loop.is_closed():
            loop.close()


@app.command()
def list_providers():
    """List available LLM providers"""
    providers = env.get_available_providers()
    if providers:
        typer.echo("Available LLM providers:")
        for provider in providers:
            config = env.get_provider_config(provider)
            typer.echo(f"  - {provider} (model: {config.model})")
    else:
        typer.echo("No LLM providers are configured. Please check your .env file.")


@app.command()
def version():
    """Show the version of the impedance agent"""
    typer.echo("Impedance Agent v0.1.0")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
