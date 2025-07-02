"""
Configuration manager for the query prediction experimentation framework.
Using Pydantic for validation of configuration settings.
"""

import yaml
import os
import json
from typing import Dict, Any, List, Optional, Union, Literal
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator
import logging

logger = logging.getLogger(__name__)


# Define Pydantic models for configuration sections
class DiscretizationConfig(BaseModel):
    enabled: bool = True
    method: Literal["equal_width", "equal_freq", "kmeans"] = "equal_width"
    bins: int = Field(5, gt=0)
    save_params: bool = True
    params_path: str = "discretization_params.pkl"


class AssociationRulesConfig(BaseModel):
    enabled: bool = True
    min_support: float = Field(0.1, ge=0.0, le=1.0)
    metric: Literal["support", "confidence", "lift", "leverage", "conviction"] = "confidence"
    min_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_len: Optional[int] = None
    
    @validator("max_len")
    def validate_max_len(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_len must be positive if specified")
        return v


class SummariesConfig(BaseModel):
    enabled: bool = True
    desired_size: int = Field(5, gt=0)
    weights: Optional[Dict[str, float]] = None


class InterestingnessConfig(BaseModel):
    enabled: bool = True
    measures: List[str] = ["variance", "simpson", "shannon"]
    
    @validator("measures")
    def validate_measures(cls, v):
        valid_measures = {"variance", "simpson", "shannon", "total", "max", "mcintosh", "gini"}
        for measure in v:
            if measure not in valid_measures:
                raise ValueError(f"Invalid interestingness measure: {measure}")
        return v


class RecommendationConfig(BaseModel):
    enabled: bool = True
    method: Literal["association_rules", "summaries", "hybrid"] = "association_rules"
    top_k: int = Field(10, gt=0)
    score_threshold: float = Field(0.5, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    metrics: List[str] = ["accuracy", "overlap", "jaccard", "precision", "recall", "f1"]
    jaccard_threshold: float = Field(0.5, ge=0.0, le=1.0)
    column_weights: Optional[Dict[str, float]] = None
    
    @validator("metrics")
    def validate_metrics(cls, v):
        valid_metrics = {
            "accuracy", "overlap", "jaccard", "precision", "recall", "f1", 
            "jaccard_precision", "jaccard_recall", "jaccard_f1"
        }
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid evaluation metric: {metric}")
        return v


class OutputConfig(BaseModel):
    save_results: bool = True
    results_dir: str = "experiment_results"
    save_format: Literal["pkl", "csv", "json"] = "pkl"


class ExperimentSettingsConfig(BaseModel):
    name: str = "default_experiment"
    prediction_gap: int = Field(1, ge=1)
    random_seed: int = 42
    sessions_limit: Optional[int] = None
    
    @validator("sessions_limit")
    def validate_sessions_limit(cls, v):
        if v is not None and v <= 0:
            raise ValueError("sessions_limit must be positive if specified")
        return v


class ExperimentConfig(BaseModel):
    experiment: ExperimentSettingsConfig = ExperimentSettingsConfig()
    discretization: DiscretizationConfig = DiscretizationConfig()
    association_rules: AssociationRulesConfig = AssociationRulesConfig()
    summaries: SummariesConfig = SummariesConfig()
    interestingness: InterestingnessConfig = InterestingnessConfig()
    recommendation: RecommendationConfig = RecommendationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    output: OutputConfig = OutputConfig()


class ConfigManager:
    """
    Manages the configuration settings for the query prediction experimentation framework.
    Handles loading config files, validating settings using Pydantic, and providing default values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        # Start with default config from Pydantic model
        self.config_model = ExperimentConfig()
        self.config = self.config_model.model_dump()
        
        # Load custom config if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a file and validate with Pydantic.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        
        Raises:
            FileNotFoundError: If the config file doesn't exist
            ValueError: If the config file format is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration from file
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
        
        # Validate and update the configuration using Pydantic
        try:
            # Create a new config model with the loaded values
            self.config_model = ExperimentConfig(**user_config)
            
            # Update the config dictionary
            self.config = self.config_model.model_dump()
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration.
        
        Returns:
            The complete configuration dictionary
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section: Name of the configuration section
            
        Returns:
            Configuration for the specified section
            
        Raises:
            KeyError: If the section doesn't exist
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        
        return self.config[section]
    
    def is_enabled(self, component: str) -> bool:
        """
        Check if a specific component is enabled.
        
        Args:
            component: Name of the component to check
            
        Returns:
            True if the component is enabled, False otherwise
        """
        return self.config.get(component, {}).get('enabled', False)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration with new values and validate.
        
        Args:
            new_config: New configuration values to apply
            
        Raises:
            ValueError: If the new configuration is invalid
        """
        # Create a merged config
        merged_config = {**self.config, **new_config}
        
        # Validate with Pydantic
        self.config_model = ExperimentConfig(**merged_config)
        
        # Update the config dictionary
        self.config = self.config_model.model_dump()
    
    def save_config(self, output_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if output_path.suffix.lower() in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif output_path.suffix.lower() == '.json':
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {output_path.suffix}")
