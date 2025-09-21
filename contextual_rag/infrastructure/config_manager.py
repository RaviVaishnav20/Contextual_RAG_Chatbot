import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigManager:
    """
    Lightweight configuration manager compatible with the external project's API.
    Loads from configs/config.yaml at project root, supports env overrides and
    auto-detects docker for Ollama host selection.
    """

    def __init__(self, config_path: str | Path | None = None):
        self.project_root = Path(__file__).resolve().parents[2]
        default_path = self.project_root / "configs" / "config.yaml"
        self.config_path = Path(config_path) if config_path else default_path
        self.config = self._load_config()

        # Detect Docker
        self.in_docker = os.path.exists("/.dockerenv")
        if self.in_docker:
            self.ollama_host = os.getenv("OLLAMA_DOCKER_HOST", "http://ollama:11434")
        else:
            self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def get_ollama_host(self) -> str:
        return self.ollama_host

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            # Return empty config if optional file missing to avoid hard failure
            return {}
        with open(self.config_path, "r") as f:
            content = f.read()
            # Substitute environment variables in the YAML content
            content = self._substitute_env_vars(content)
            return yaml.safe_load(content) or {}
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in the format ${VAR:-default}"""
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern to match ${VAR:-default} or ${VAR}
        pattern = r'\$\{([^:}]+)(?::-([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value: Any = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_database_config(self) -> Dict[str, Any]:
        return self.config.get("database", {})

    def get_chunking_config(self) -> Dict[str, Any]:
        return self.config.get("chunking", {})

    def get_contextual_config(self) -> Dict[str, Any]:
        return self.config.get("contextual_retrieval", {})

    # def get_reranker_config(self) -> Dict[str, Any]:
    #     return self.config.get("reranker", {})

    def get_embedding_config(self) -> Dict[str, Any]:
        return self.config.get("embedding", {})

    def get_rag_config(self) -> Dict[str, Any]:
        return self.config.get("rag", {})

    def get_crewai_config(self) -> Dict[str, Any]:
        return self.config.get("crewai", {})

    def get_directories(self) -> Dict[str, str]:
        return self.config.get("directories", {})

    def get_crewai_config(self) -> Dict[str, Any]:
        return self.config.get("crewai", {})

    def get_agents_config(self) -> Dict[str, Any]:
        agents_config_path = self.project_root / "configs" / "crew" / "agents.yaml"
        return self._load_specific_config(agents_config_path)

    def get_tasks_config(self) -> Dict[str, Any]:
        tasks_config_path = self.project_root / "configs" / "crew" / "tasks.yaml"
        return self._load_specific_config(tasks_config_path)

    def _load_specific_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            print(f"File not found: {path}")
            return {}
        with open(path, "r") as f:
            content = f.read()
            content = self._substitute_env_vars(content)
            return yaml.safe_load(content) or {}

    # def get_directories(self) -> Dict[str, str]:
    #     return self.config.get("directories", {})


