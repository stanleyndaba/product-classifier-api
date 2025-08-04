"""
Configuration parser for the Smart Product Classifier.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


class ConfigParser:
    """Configuration parser that handles YAML configs and environment variables."""
    
    def __init__(self, config_path: str = "config/config.yaml", env_file: str = ".env"):
        self.config_path = Path(config_path)
        self.env_file = Path(env_file)
        self._config = None
        
        # Load environment variables
        if self.env_file.exists():
            load_dotenv(self.env_file)
    
    def get_config(self, environment: str = None) -> Dict[str, Any]:
        """
        Get configuration for specified environment.
        
        Args:
            environment: Environment name (development, production, training)
                        If None, uses ENV environment variable or defaults to development
                        
        Returns:
            Configuration dictionary
        """
        if self._config is None:
            self._load_config()
        
        if environment is None:
            environment = os.getenv('ENV', 'development')
        
        # Return environment-specific config or fallback to development
        return self._config.get(environment, self._config.get('development', {}))
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            # Create default config if file doesn't exist
            self._config = self._get_default_config()
            self._save_default_config()
        else:
            try:
                with open(self.config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            except Exception as e:
                print(f"Error loading config from {self.config_path}: {e}")
                self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'development': {
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'debug': True,
                    'reload': True,
                    'workers': 1
                },
                'model': {
                    'path': 'saved_models/best_model.pth',
                    'device': 'cpu',
                    'confidence_threshold': 0.5,
                    'batch_size_limit': 20,
                    'classes': [
                        'electronics', 'clothing', 'furniture', 'books',
                        'toys', 'sports', 'beauty', 'home_garden',
                        'automotive', 'health', 'food', 'jewelry'
                    ]
                },
                'data': {
                    'max_file_size': 10485760,  # 10MB
                    'allowed_extensions': ['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                    'image_size': [224, 224]
                },
                'rate_limiting': {
                    'enabled': False,
                    'requests_per_minute': 100
                },
                'logging': {
                    'level': 'DEBUG',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
                'redis': {
                    'enabled': False,
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                }
            }
        }
    
    def _save_default_config(self) -> None:
        """Save default configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save default config to {self.config_path}: {e}")
    
    def get_env_var(self, key: str, default: Any = None, var_type: type = str) -> Any:
        """
        Get environment variable with type conversion.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            var_type: Type to convert to (str, int, float, bool)
            
        Returns:
            Environment variable value with type conversion
        """
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        if var_type == bool:
            return str(value).lower() in ('true', '1', 'yes', 'on')
        elif var_type == int:
            try:
                return int(value)
            except ValueError:
                return default
        elif var_type == float:
            try:
                return float(value)
            except ValueError:
                return default
        else:
            return str(value)
    
    def override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override configuration values with environment variables.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment variable overrides
        """
        # Override common settings with environment variables
        env_mappings = {
            'API_HOST': ('api', 'host'),
            'API_PORT': ('api', 'port'),
            'API_DEBUG': ('api', 'debug'),
            'API_RELOAD': ('api', 'reload'),
            'MODEL_PATH': ('model', 'path'),
            'MODEL_DEVICE': ('model', 'device'),
            'MODEL_CONFIDENCE_THRESHOLD': ('model', 'confidence_threshold'),
            'BATCH_SIZE_LIMIT': ('model', 'batch_size_limit'),
            'RATE_LIMIT_REQUESTS': ('rate_limiting', 'requests_per_minute'),
            'RATE_LIMIT_ENABLED': ('rate_limiting', 'enabled'),
            'LOG_LEVEL': ('logging', 'level'),
            'REDIS_HOST': ('redis', 'host'),
            'REDIS_PORT': ('redis', 'port'),
            'REDIS_ENABLED': ('redis', 'enabled'),
        }
        
        result = config.copy()
        
        for env_key, (section, key) in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                if section not in result:
                    result[section] = {}
                
                # Convert type based on existing value or reasonable defaults
                existing_value = result[section].get(key)
                if isinstance(existing_value, bool):
                    result[section][key] = env_value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(existing_value, int):
                    try:
                        result[section][key] = int(env_value)
                    except ValueError:
                        pass
                elif isinstance(existing_value, float):
                    try:
                        result[section][key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    result[section][key] = env_value
        
        return result
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_sections = ['api', 'model', 'data']
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        # Validate API section
        api_config = config.get('api', {})
        if not isinstance(api_config.get('port'), int):
            print("API port must be an integer")
            return False
        
        # Validate model section
        model_config = config.get('model', {})
        if not model_config.get('classes'):
            print("Model classes list cannot be empty")
            return False
        
        return True