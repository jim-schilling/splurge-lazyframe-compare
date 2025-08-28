"""Configuration and settings helpers for the comparison framework."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl


class ConfigConstants:
    """Constants for configuration management."""

    # Environment variable prefixes
    ENV_PREFIX: str = "SPLURGE_"

    # Default configuration file names
    DEFAULT_CONFIG_FILE: str = "comparison_config.json"
    DEFAULT_SCHEMA_FILE: str = "schemas.json"

    # Configuration sections
    COMPARISON_SECTION: str = "comparison"
    SCHEMA_SECTION: str = "schemas"
    REPORTING_SECTION: str = "reporting"
    PERFORMANCE_SECTION: str = "performance"

    # Configuration validation
    REQUIRED_CONFIG_KEYS: list = ["primary_key_columns", "column_mappings"]
    VALID_NULL_POLICIES: list = ["equals", "not_equals", "ignore"]
    VALID_CASE_POLICIES: list = ["sensitive", "insensitive", "preserve"]


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}") from e


def save_config_to_file(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to a JSON file.

    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    except Exception as e:
        raise ValueError(f"Failed to save configuration: {e}") from e


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.

    Args:
        base_config: Base configuration.
        override_config: Override configuration.

    Returns:
        Merged configuration dictionary.
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override the value
            merged[key] = value

    return merged


def get_env_config(prefix: str = ConfigConstants.ENV_PREFIX) -> Dict[str, Any]:
    """Load configuration from environment variables.

    Args:
        prefix: Environment variable prefix to filter by.

    Returns:
        Configuration dictionary from environment variables.
    """
    config = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase with underscores
            config_key = key[len(prefix):].lower().replace('_', '.')

            # Try to parse as JSON first, then as primitive types
            try:
                import json
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    parsed_value = value.lower() == 'true'
                elif value.isdigit():
                    parsed_value = int(value)
                elif value.replace('.', '').isdigit():
                    parsed_value = float(value)
                else:
                    parsed_value = value

            # Set nested dictionary value
            set_nested_config_value(config, config_key, parsed_value)

    return config


def set_nested_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a value in a nested configuration dictionary using dot notation.

    Args:
        config: Configuration dictionary to modify.
        key_path: Dot-separated path to the configuration key.
        value: Value to set.
    """
    keys = key_path.split('.')
    current = config

    # Navigate to the correct nested level
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def validate_config(config: Dict[str, Any]) -> list[str]:
    """Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        List of validation error messages. Empty if valid.
    """
    errors = []

    # Check required keys
    for required_key in ConfigConstants.REQUIRED_CONFIG_KEYS:
        if required_key not in config:
            errors.append(f"Missing required configuration key: {required_key}")

    # Validate primary key columns
    if "primary_key_columns" in config:
        pk_cols = config["primary_key_columns"]
        if not isinstance(pk_cols, list) or not pk_cols:
            errors.append("primary_key_columns must be a non-empty list")
        elif not all(isinstance(col, str) for col in pk_cols):
            errors.append("All primary key columns must be strings")

    # Validate column mappings
    if "column_mappings" in config:
        mappings = config["column_mappings"]
        if not isinstance(mappings, list):
            errors.append("column_mappings must be a list")
        else:
            for i, mapping in enumerate(mappings):
                if not isinstance(mapping, dict):
                    errors.append(f"column_mappings[{i}] must be a dictionary")
                elif not all(key in mapping for key in ["left_column", "right_column", "comparison_name"]):
                    errors.append(f"column_mappings[{i}] missing required keys")

    # Validate null policy
    if "null_equals_null" in config:
        null_policy = config["null_equals_null"]
        if not isinstance(null_policy, bool):
            errors.append("null_equals_null must be a boolean")

    # Validate case sensitivity
    if "ignore_case" in config:
        case_policy = config["ignore_case"]
        if not isinstance(case_policy, bool):
            errors.append("ignore_case must be a boolean")

    # Validate tolerance
    if "tolerance" in config:
        tolerance = config["tolerance"]
        if not isinstance(tolerance, dict):
            errors.append("tolerance must be a dictionary")
        else:
            for col, tol in tolerance.items():
                if not isinstance(tol, (int, float)) or tol <= 0:
                    errors.append(f"tolerance for column '{col}' must be a positive number")

    return errors


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration template.

    Returns:
        Default configuration dictionary.
    """
    return {
        "primary_key_columns": ["id"],
        "column_mappings": [
            {
                "left_column": "id",
                "right_column": "customer_id",
                "comparison_name": "id"
            }
        ],
        "ignore_case": False,
        "null_equals_null": True,
        "tolerance": {},
        "reporting": {
            "max_samples": 10,
            "table_format": "grid",
            "include_summary": True,
            "include_differences": True,
        },
        "performance": {
            "batch_size": 10000,
            "parallel_workers": 4,
            "memory_limit_mb": 1000,
        },
    }


def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Args:
        config: Base configuration dictionary.

    Returns:
        Configuration with environment overrides applied.
    """
    env_config = get_env_config()
    if env_config:
        return merge_configs(config, env_config)
    return config


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default_value: Any = None
) -> Any:
    """Get a value from nested configuration using dot notation.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the configuration key.
        default_value: Default value if key not found.

    Returns:
        Configuration value or default value.
    """
    keys = key_path.split('.')
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default_value


def create_config_from_dataframes(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    primary_keys: list[str],
    auto_map_columns: bool = True
) -> Dict[str, Any]:
    """Create a configuration from DataFrame schemas.

    Args:
        left_df: Left LazyFrame.
        right_df: Right LazyFrame.
        primary_keys: List of primary key column names.
        auto_map_columns: Whether to auto-map columns with same names.

    Returns:
        Generated configuration dictionary.
    """
    left_schema = left_df.collect_schema()
    right_schema = right_df.collect_schema()

    # Create column mappings
    mappings = []
    left_columns = set(left_schema.names())
    right_columns = set(right_schema.names())

    if auto_map_columns:
        # Auto-map columns with same names
        common_columns = left_columns & right_columns
        for col in common_columns:
            mappings.append({
                "left_column": col,
                "right_column": col,
                "comparison_name": col
            })

        # Add remaining columns as separate mappings
        for col in left_columns - common_columns:
            mappings.append({
                "left_column": col,
                "right_column": col,  # Same name for simplicity
                "comparison_name": col
            })
    else:
        # Use all columns from left as base
        for col in left_columns:
            mappings.append({
                "left_column": col,
                "right_column": col,
                "comparison_name": col
            })

    return {
        "primary_key_columns": primary_keys,
        "column_mappings": mappings,
        "ignore_case": False,
        "null_equals_null": True,
        "tolerance": {},
    }
