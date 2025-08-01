"""
Application configuration management.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # Server Configuration
    app_name: str = Field(default="Smart Product Classifier API", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Security Configuration
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["GET", "POST"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    # Model Configuration
    model_path: str = Field(default="models/product_classifier.pt", env="MODEL_PATH")
    model_device: str = Field(default="cpu", env="MODEL_DEVICE")  # cpu or cuda
    model_cache_size: int = Field(default=1, env="MODEL_CACHE_SIZE")
    
    # Image Processing Configuration
    max_image_size: int = Field(default=10 * 1024 * 1024, env="MAX_IMAGE_SIZE")  # 10MB
    allowed_image_types: List[str] = Field(
        default=["image/jpeg", "image/png", "image/webp"], 
        env="ALLOWED_IMAGE_TYPES"
    )
    image_resize_dimensions: tuple = Field(default=(224, 224), env="IMAGE_RESIZE_DIMENSIONS")
    
    # Rate Limiting Configuration
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Batch Processing Configuration
    max_batch_size: int = Field(default=20, env="MAX_BATCH_SIZE")
    min_batch_size: int = Field(default=2, env="MIN_BATCH_SIZE")
    batch_timeout: int = Field(default=30, env="BATCH_TIMEOUT")  # seconds
    
    # Database Configuration (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Performance Configuration
    worker_processes: int = Field(default=1, env="WORKER_PROCESSES")
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is supported."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('model_device')
    def validate_model_device(cls, v):
        """Validate model device is supported."""
        valid_devices = ['cpu', 'cuda', 'mps']
        if v.lower() not in valid_devices:
            raise ValueError(f'Model device must be one of: {valid_devices}')
        return v.lower()
    
    @validator('allowed_hosts', 'cors_origins', pre=True)
    def parse_list_from_string(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_enabled: bool = False


class ProductionSettings(Settings):
    """Production-specific settings."""
    debug: bool = False
    log_level: str = "INFO"
    rate_limit_enabled: bool = True
    allowed_hosts: List[str] = []  # Should be set via environment


class TestingSettings(Settings):
    """Testing-specific settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    rate_limit_enabled: bool = False
    model_path: str = "tests/fixtures/test_model.pt"


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Settings: Application settings instance
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


def get_model_config() -> dict:
    """
    Get model-specific configuration.
    
    Returns:
        dict: Model configuration parameters
    """
    settings = get_settings()
    
    return {
        "model_path": settings.model_path,
        "device": settings.model_device,
        "cache_size": settings.model_cache_size,
        "image_dimensions": settings.image_resize_dimensions,
        "categories": [
            "electronics", "clothing", "furniture", "books", "toys",
            "sports", "beauty", "automotive", "home", "garden",
            "jewelry", "food"
        ]
    }


def get_redis_config() -> Optional[dict]:
    """
    Get Redis configuration if available.
    
    Returns:
        Optional[dict]: Redis configuration or None
    """
    settings = get_settings()
    
    if settings.redis_url:
        return {
            "url": settings.redis_url,
            "encoding": "utf-8",
            "decode_responses": True
        }
    
    return None