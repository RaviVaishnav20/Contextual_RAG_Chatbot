
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration container."""
    host: str
    port: int
    database: str
    username: str
    password: str
    table_name: str
    driver: str = "psycopg2"
    ssl_mode: str = "prefer"
    
    @property
    def dsn(self) -> str:
        """Generate PostgreSQL DSN from configuration."""
        # Handle environment variable substitution
        host = self._substitute_env_vars(self.host)
        port = self._substitute_env_vars(str(self.port))
        database = self._substitute_env_vars(self.database)
        username = self._substitute_env_vars(self.username)
        password = self._substitute_env_vars(self.password)
        
        return f"postgresql+{self.driver}://{username}:{password}@{host}:{port}/{database}?sslmode={self.ssl_mode}"
    
    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in string values."""
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value


class ConfigLoader:
    """Configuration loader for YAML-based configuration files."""
    
    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(__file__).resolve().parents[2] / "configs"
    
    def load_database_config(self, environment: str | None = None) -> DatabaseConfig:
        """Load database configuration for the specified environment."""
        env = environment or os.getenv("ENVIRONMENT", "default")
        
        config_file = self.config_dir / "database.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in configuration file")
        
        db_config = config_data[env]["postgres"]
        return DatabaseConfig(**db_config)
    
    def get_postgres_dsn(self, environment: str | None = None) -> str:
        """Get PostgreSQL DSN for the specified environment."""
        db_config = self.load_database_config(environment)
        return db_config.dsn


# Global configuration loader instance
config_loader = ConfigLoader()
