# src/core/config.py
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, model_validator
import yaml
from pathlib import Path
from .exceptions import ConfigError


class Variable(BaseModel):
    name: str
    initialValue: float
    lowerBound: float
    upperBound: float

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v

    @validator("initialValue")
    def validate_bounds(cls, v, values):
        if "lowerBound" in values and "upperBound" in values:
            if not (values["lowerBound"] <= v <= values["upperBound"]):
                raise ValueError(
                    f'Initial value {v} must be within bounds [{values["lowerBound"]}, {values["upperBound"]}]'
                )
        return v


class WeightingConfig(BaseModel):
    type: str = "modulus"
    data: Dict[str, Any] = Field(default_factory=dict)

    @validator("type")
    def validate_type(cls, v):
        allowed_types = ["modulus", "proportional", "unit", "sigma"]
        if v not in allowed_types:
            raise ValueError(f"Weighting type must be one of: {allowed_types}")
        return v.lower()


class FitterConfig(BaseModel):
    model_code: str
    variables: List[Variable]
    weighting: WeightingConfig = Field(default_factory=lambda: WeightingConfig())

    @validator("model_code")
    def validate_model_code(cls, v):
        if not v.strip():
            raise ValueError("Model code cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_variables_unique(cls, values):
        names = [var.name for var in values.variables]
        if len(names) != len(set(names)):
            raise ValueError("Variable names must be unique")
        return values


class Config:
    def __init__(self, config_path: str = None, config_dict: Dict = None):
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = config_dict or {}

        try:
            self.fitter = FitterConfig(**config_data)
        except ValueError as e:
            raise ConfigError(f"Invalid configuration: {str(e)}")

    @classmethod
    def load_model(cls, config_path: Union[str, Path]) -> Dict:
        """Load model configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            return config_data
        except Exception as e:
            raise ConfigError(f"Failed to load model config: {str(e)}")
