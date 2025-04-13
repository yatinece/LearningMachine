# LearnAI Ready

A machine learning experiment management framework that simplifies the process of configuring, executing, and tracking ML experiments.

## Features

- **Configuration Management**: Easy-to-use configuration system for ML experiments
- **Flexible Data Sources**: Support for both local and Ceph storage
- **Model Selection**: Built-in support for XGBoost and CatBoost
- **Hyperparameter Tuning**: Automated hyperparameter optimization
- **Experiment Tracking**: Integration with MLflow for experiment tracking
- **REST API**: Easy-to-use API for configuration and experiment management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/learnai_ready.git
cd learnai_ready
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

1. Start the API server:
```bash
uvicorn learnai_ready.api.app:app --reload
```

2. Access the API documentation at `http://127.0.0.1:8000/docs`

3. Create your first configuration:
```python
from learnai_ready.core.services import ConfigService

# Initialize the config service
config_service = ConfigService()

# Create a default configuration
config = config_service.create_full_config()

# Generate YAML
yaml_str = config_service.generate_yaml(config.dict())

# Save to file
file_path = config_service.save_yaml(config.dict())
```

## API Endpoints

- `GET /`: Welcome message
- `POST /api/config/generate`: Generate a new configuration
- `GET /api/config/templates/{section}`: Get configuration templates
- `POST /api/config/validate`: Validate a configuration
- `POST /api/config/upload`: Upload a configuration file
- `GET /api/config/list`: List available configurations
- `GET /api/config/download/{config_id}`: Download a configuration
- `POST /api/execute`: Execute ML pipeline tasks
- `GET /api/available-tasks/{config_id}`: Get available tasks

## Configuration Structure

The configuration is divided into several sections:

1. **Data Source**: Configure data input sources (local or Ceph)
2. **Model Settings**: Define target variable and features
3. **Model Selection**: Choose and configure ML models
4. **Hyperparameter Tuning**: Set up optimization parameters
5. **Training**: Configure training process
6. **Experiment Tracking**: Set up experiment tracking
7. **Output**: Configure output formats and locations

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
