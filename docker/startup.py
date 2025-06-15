import logging
import sys
from typing import Dict
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yml") -> Dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        Dict: Configuration settings.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

def ensure_ollama_model_available(ollama_base_url: str, ollama_model: str) -> bool:
    """
    Check if Ollama server is running and the model is available, pulling it if necessary.
    
    Args:
        ollama_base_url: Base URL for the Ollama service
        ollama_model: Name of the model to check/pull
        
    Returns:
        bool: True if model is available and ready
    """
    logger.info(f"Checking Ollama status at {ollama_base_url} and model '{ollama_model}'")
    
    try:
        response = requests.get(f"{ollama_base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]
        
        if ollama_model in model_names:
            logger.info(f"Ollama model '{ollama_model}' found.")
            return True

        logger.warning(f"Ollama model '{ollama_model}' not found. Attempting to pull...")
        
        pull_response = requests.post(
            f"{ollama_base_url}/api/pull", 
            json={"name": ollama_model, "stream": False},
            timeout=600  # Allow 10 minutes for download
        )
        pull_response.raise_for_status() 
        
        pull_data = pull_response.json()
        if pull_data.get("status", "").lower() == "success":
            logger.info(f"Successfully pulled Ollama model '{ollama_model}'.")
            return True
        else:
            logger.warning(f"Pulled Ollama model '{ollama_model}', response status: {pull_data.get('status', 'N/A')}. Assuming success.")
            return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to or communicating with Ollama at {ollama_base_url}: {e}")
        logger.error("Please ensure the Ollama service is running and accessible.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while ensuring Ollama model availability: {e}")
        sys.exit(1)

def initialize_ollama(config_path: str = "config.yml") -> Dict:
    """
    Initialize the Ollama service with configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
    
    Returns:
        Dict: Configuration settings
    """
    logger.info("Initializing Ollama service")
    config = load_config(config_path)
    
    # Load Ollama config
    ollama_config = config.get("ollama", {})
    ollama_base_url = ollama_config.get("base_url")
    ollama_model = ollama_config.get("model")
    
    if not all([ollama_base_url, ollama_model]):
        logger.error("Incomplete Ollama configuration in config.yaml (missing base_url or model)")
        sys.exit(1)
    
    # Ensure Ollama model is ready
    ensure_ollama_model_available(ollama_base_url, ollama_model)
    
    return config

if __name__ == "__main__":
    config = initialize_ollama()
    logger.info("Ollama service initialized successfully")