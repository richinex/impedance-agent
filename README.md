# Impedance Analysis Tool

An AI-powered tool for analyzing electrochemical impedance spectroscopy (EIS) data using Distribution of Relaxation Times (DRT), Equivalent Circuit Modeling (ECM), and Lin-KK validation.

## Features

- **Distribution of Relaxation Times (DRT) Analysis**
- **Equivalent Circuit Model (ECM) Fitting**
- **Lin-KK Data Validation**
- **AI-Assisted Interpretation of Results**
- **Multiple Output Formats**: JSON, CSV, Excel
- **Configurable Logging and Debug Modes**

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.env\Scripts\activate  # Windows

# Install package
pip install -e .

# Setup environment variables
cp impedance_agent/.env.example .env
# Edit .env with your API keys
```

## Usage

### Basic Usage

```bash
impedance-agent data/impedance.txt
```

### With ECM Fitting

```bash
impedance-agent data/impedance.txt --ecm configs/models/randles.yaml
```

### Full Options

To run the examples use

```bash
python -m src.cli.main \
  examples/data/impedance.txt \
  --ecm examples/models/randles.yaml \
  --output-path results/analysis.json \
  --output-format json \
  --plot-format png \
  --plot \
  --log-level DEBUG \
  --debug
```

### Command Line Options

- `data_path`: Path to impedance data file (required)
- `--ecm`: Path to the equivalent circuit model (ECM) configuration file
- `--output-path`: Path for saving results
- `--output-format`: Output format (json/csv/excel)
- `--plot-format`: Plot format (png/pdf/svg)
- `--plot`: Generate plots (default: True)
- `--show-plots`: Display plots in window (default: False)
- `--log-level`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `--debug`: Enable debug mode (default: False)
- `--workers`: Number of worker processes (default: auto-detected)

### Additional Commands

- `list_providers`: List available LLM providers.
- `version`: Show the version of the impedance agent.

## Input Data Format

Supports CSV/TXT files with the following columns:

- `frequency` (Hz)
- `Z_real` (Ω)
- `Z_imag` (Ω)

## Model Configuration

YAML format for ECM definition:

```yaml
model_code: |
  def impedance_model(p, f):
    w = 2 * jnp.pi * f
    Rs, Rct, Cdl = p
    Z = Rs + Rct / (1 + 1j * w * Cdl * Rct)
    return jnp.concatenate([Z.real, Z.imag])

variables:
  - name: Rs
    initialValue: 0.1
    lowerBound: 1e-6
    upperBound: 1000.0
  - name: Rct
    initialValue: 1.0
    lowerBound: 1e-6
    upperBound: 1e6
  - name: Cdl
    initialValue: 1e-6
    lowerBound: 1e-12
    upperBound: 1e-3
```

## Output

The tool provides:

- DRT Analysis Results
- ECM Fitting Parameters (if ECM provided)
- Lin-KK Validation Metrics
- AI-Generated Interpretation
- Detailed Recommendations

## Requirements

- Python 3.9+
- JAX/JAXopt
- NumPy/SciPy
- Pydantic
- OpenAI API Access
- `impedance.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{impedance_agent,
  author = {Chukwu, Richard},
  title = {Impedance-Agent: AI-powered EIS Analysis Tool},
  year = {2024},
  url = {https://github.com/richinex/impedance-agent}
}
```

## Support the Project

If you find this tool useful, consider buying me a coffee:

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/YOUR_USERNAME)

## Documentation

For detailed documentation, visit: [Documentation Link]

## Acknowledgments

This project uses several open-source packages including:

- JAX/JAXopt for optimization
- `impedance.py` for impedance analysis
- OpenAI's API for AI-assisted interpretation

## License

This project is licensed under the MIT License.

Made with ❤️ by Richard Chukwu
