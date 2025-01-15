# src/core/exporters.py
from pathlib import Path
from typing import Union
import json
import pandas as pd
import numpy as np
from .models import AnalysisResult
from .exceptions import ExportError
import logging
import aiofiles
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io


class ResultExporter:
    """Handles exporting analysis results to various formats with async I/O"""

    logger = logging.getLogger(__name__)

    @staticmethod
    async def export_async(
        result: AnalysisResult, file_path: Union[str, Path], format: str = "json"
    ) -> None:
        """Async export of analysis results to file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            ResultExporter.logger.info(f"Exporting results to {file_path}")

            export_funcs = {
                "json": ResultExporter._export_json_async,
                "csv": ResultExporter._export_csv_async,
                "excel": ResultExporter._export_excel_async,
                "md": ResultExporter._export_markdown_async,
            }

            if format not in export_funcs:
                raise ExportError(f"Unsupported export format: {format}")

            await export_funcs[format](result, file_path)
            ResultExporter.logger.info("Export complete")

        except (IOError, OSError) as e:
            raise ExportError(f"Failed to write file: {str(e)}")
        except Exception as e:
            if not isinstance(e, ExportError):
                raise ExportError(f"Error during export: {str(e)}")
            raise

    @staticmethod
    def export(
        result: AnalysisResult, file_path: Union[str, Path], format: str = "json"
    ) -> None:
        """Synchronous wrapper for async export"""
        asyncio.run(ResultExporter.export_async(result, file_path, format))

    @staticmethod
    async def _export_json_async(result: AnalysisResult, file_path: Path) -> None:
        """Async JSON export with streaming"""
        try:
            data = result.model_dump()

            # Use a buffer for large datasets
            buffer = io.StringIO()
            json.dump(data, buffer, indent=2, default=ResultExporter._json_serialize)

            async with aiofiles.open(file_path, "w") as f:
                await f.write(buffer.getvalue())
            buffer.close()
        except Exception as e:
            raise ExportError(f"JSON export failed: {str(e)}")

    @staticmethod
    async def _export_csv_async(result: AnalysisResult, file_path: Path) -> None:
        """Async CSV export with parallel processing"""
        try:
            base_path = file_path.parent / file_path.stem
            tasks = []

            # Prepare export tasks
            if result.ecm_fit:
                tasks.append(
                    ResultExporter._write_ecm_csv(result, f"{base_path}_ecm.csv")
                )

            if result.drt_fit:
                tasks.append(
                    ResultExporter._write_drt_csv(result, f"{base_path}_drt.csv")
                )

            # Always write summary
            tasks.append(
                ResultExporter._write_summary(result, f"{base_path}_summary.txt")
            )

            # Execute all file writes concurrently
            await asyncio.gather(*tasks)
        except Exception as e:
            raise ExportError(f"CSV export failed: {str(e)}")

    @staticmethod
    async def _export_excel_async(result: AnalysisResult, file_path: Path) -> None:
        """Async Excel export with optimized memory usage"""
        try:
            # Create Excel writer in memory first
            buffer = io.BytesIO()

            def write_excel():
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    if result.ecm_fit:
                        pd.DataFrame(
                            {
                                "Parameter": [
                                    p["name"] for p in result.ecm_fit.param_info
                                ],
                                "Value": result.ecm_fit.parameters,
                                "Error": result.ecm_fit.errors,
                            }
                        ).to_excel(writer, sheet_name="ECM_Fit", index=False)

                    if result.drt_fit:
                        pd.DataFrame(
                            {
                                "tau": result.drt_fit.tau,
                                "gamma": ResultExporter._process_gamma(
                                    result.drt_fit.gamma
                                ),
                            }
                        ).to_excel(writer, sheet_name="DRT_Results", index=False)

                    # Always write summary
                    pd.DataFrame(
                        {
                            "Summary": [result.summary],
                            "Recommendations": ["\n".join(result.recommendations)],
                        }
                    ).to_excel(writer, sheet_name="Summary", index=False)

            # Run Excel writing in thread pool
            with ThreadPoolExecutor() as pool:
                await asyncio.get_event_loop().run_in_executor(pool, write_excel)

            # Write buffer to file asynchronously
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(buffer.getvalue())
            buffer.close()
        except Exception as e:
            raise ExportError(f"Excel export failed: {str(e)}")

    @staticmethod
    async def _write_ecm_csv(result: AnalysisResult, filepath: str) -> None:
        """Helper for async ECM CSV writing"""
        df = pd.DataFrame(
            {
                "Parameter": [p["name"] for p in result.ecm_fit.param_info],
                "Value": result.ecm_fit.parameters,
                "Error": result.ecm_fit.errors,
            }
        )
        async with aiofiles.open(filepath, "w") as f:
            await f.write(df.to_csv(index=False))

    @staticmethod
    async def _write_drt_csv(result: AnalysisResult, filepath: str) -> None:
        """Helper for async DRT CSV writing"""
        data = {
            "tau": result.drt_fit.tau,
            "gamma": ResultExporter._process_gamma(result.drt_fit.gamma),
        }
        df = pd.DataFrame(data)
        async with aiofiles.open(filepath, "w") as f:
            await f.write(df.to_csv(index=False))

    @staticmethod
    async def _write_summary(result: AnalysisResult, filepath: str) -> None:
        """Helper for async summary writing"""
        async with aiofiles.open(filepath, "w") as f:
            await f.write(result.summary)

    @staticmethod
    def _process_gamma(gamma):
        """Process gamma values for export"""
        if isinstance(gamma, complex):
            return pd.Series({"gamma_real": gamma.real, "gamma_imag": gamma.imag})
        return gamma

    @staticmethod
    def _json_serialize(obj):
        """Handle JSON serialization of special types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    @staticmethod
    async def _export_markdown_async(result: AnalysisResult, file_path: Path) -> None:
        """Async Markdown export of analysis summary"""
        try:
            markdown_content = f"""# Impedance Analysis Summary\n\n{result.summary}\n"""

            # Add recommendations
            markdown_content += "\n## Recommendations\n\n"
            if result.recommendations:
                for rec in result.recommendations:
                    markdown_content += f"* {rec}\n"

            # Add fit metrics with error handling
            if result.ecm_fit:
                markdown_content += "\n## ECM Fit Metrics\n\n"
                try:
                    markdown_content += f"* WRMS: {result.ecm_fit.wrms:.6e}\n"
                    markdown_content += f"* χ²: {result.ecm_fit.chi_square:.6e}\n"
                    markdown_content += f"* AIC: {result.ecm_fit.aic:.6e}\n"
                except AttributeError as e:
                    raise ExportError(f"Missing required fit metrics: {str(e)}")

                # Add parameter table
                markdown_content += "\n### Fitted Parameters\n\n"
                markdown_content += "| Parameter | Value | Error |\n"
                markdown_content += "|-----------|--------|--------|\n"
                for p, v, e in zip(
                    [p["name"] for p in result.ecm_fit.param_info],
                    result.ecm_fit.parameters,
                    result.ecm_fit.errors,
                ):
                    markdown_content += f"| {p} | {v:.6e} | {e:.6e} |\n"

            # Write markdown file
            async with aiofiles.open(file_path, "w") as f:
                await f.write(markdown_content)

        except (IOError, OSError) as e:
            raise ExportError(f"Failed to write markdown file: {str(e)}")
        except Exception as e:
            raise ExportError(f"Error during markdown export: {str(e)}")
