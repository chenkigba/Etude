# etude/config/loader.py

"""Configuration loading utilities."""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .schema import EtudeConfig


def _get_default_config_path() -> Optional[Path]:
    """
    Get the path to the built-in default configuration file.

    Returns:
        Path to the default config if available, None otherwise.
    """
    if sys.version_info >= (3, 9):
        from importlib.resources import files
        try:
            config_dir = files("etude.configs")
            default_yaml = config_dir.joinpath("default.yaml")
            # For Python 3.9+, we can use as_file or check if it's a Traversable
            if hasattr(default_yaml, "read_text"):
                return default_yaml
        except Exception:
            pass
    else:
        # Python 3.8 fallback
        from importlib.resources import path as resource_path
        try:
            with resource_path("etude.configs", "default.yaml") as p:
                if p.exists():
                    return p
        except Exception:
            pass

    # Fallback: try relative to this file
    fallback_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if fallback_path.exists():
        return fallback_path

    return None


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> EtudeConfig:
    """
    Load configuration with optional YAML overrides.

    Args:
        config_path: Path to YAML file with overrides (optional).
                    If None, uses built-in defaults from the package.
        overrides: Dict of programmatic overrides (optional)

    Returns:
        Complete EtudeConfig with all defaults filled in

    Example:
        # Use all defaults (package built-in)
        config = load_config()

        # Override from external YAML
        config = load_config(Path("my_config.yaml"))

        # Override programmatically
        config = load_config(overrides={"decoder": {"temperature": 0.8}})

        # Combine both
        config = load_config(
            Path("my_config.yaml"),
            overrides={"env": {"device": "cuda"}}
        )
    """
    config_dict: Dict[str, Any] = {}

    # Load built-in default config first
    default_config_path = _get_default_config_path()
    if default_config_path is not None:
        try:
            if hasattr(default_config_path, "read_text"):
                # importlib.resources Traversable
                yaml_content = default_config_path.read_text()
                yaml_config = yaml.safe_load(yaml_content) or {}
            else:
                # Regular Path
                with open(default_config_path, "r") as f:
                    yaml_config = yaml.safe_load(f) or {}
            config_dict = _deep_merge(config_dict, yaml_config)
        except Exception:
            pass  # Fall back to schema defaults

    # Load user-specified YAML overrides if provided
    if config_path is not None:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
                config_dict = _deep_merge(config_dict, yaml_config)

    # Apply programmatic overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    # Create config with validation
    return EtudeConfig(**config_dict)


def save_config(config: EtudeConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML.

    Useful for experiment tracking - saves the complete configuration
    used for a training run.

    Args:
        config: The configuration to save
        path: Output path for the YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
