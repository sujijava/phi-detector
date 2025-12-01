"""
Ollama Client for interacting with local Ollama API.
"""

import json
import logging
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    raise ImportError(
        "requests library not installed. "
        "Install with: pip install requests"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama API.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama client.

        Args:
            model: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.tags_url = f"{self.base_url}/api/tags"

        logger.info(f"Initialized OllamaClient with model: {model}")
        logger.info(f"Base URL: {self.base_url}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Optional[str]:
        """
        Generate text using Ollama API.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response

        Returns:
            Generated text response, or None if error occurs

        Raises:
            ValueError: If prompt is empty
            requests.exceptions.RequestException: If API request fails
        """
        try:
            if not prompt or not prompt.strip():
                raise ValueError("Prompt cannot be empty")

            logger.info(f"Generating response for prompt (length: {len(prompt)} chars)")

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }

            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120
            )

            # Check for HTTP errors
            response.raise_for_status()

            # Parse response
            if stream:
                # Handle streaming response
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                full_response += json_response['response']
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse streaming response: {e}")
                            continue
                return full_response
            else:
                # Handle non-streaming response
                result = response.json()

                if 'response' in result:
                    generated_text = result['response']
                    logger.info(f"Generated response (length: {len(generated_text)} chars)")
                    return generated_text
                else:
                    logger.error(f"Unexpected response format: {result}")
                    return None

        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            raise

        except requests.exceptions.Timeout:
            logger.error("Request timed out. Ollama may be processing a large request.")
            return None

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Could not connect to Ollama at {self.base_url}. "
                "Is Ollama running?"
            )
            return None

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error occurred: {e}")
            if e.response.status_code == 404:
                logger.error(
                    f"Model '{self.model}' not found. "
                    "Pull it with: ollama pull {self.model}"
                )
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return None

    def check_health(self) -> bool:
        """
        Check if Ollama service is running and accessible.

        Returns:
            True if Ollama is healthy, False otherwise
        """
        try:
            logger.info(f"Checking Ollama health at {self.base_url}")

            # Try to get list of available models
            response = requests.get(
                self.tags_url,
                timeout=5
            )

            response.raise_for_status()

            # Check if our model is available
            result = response.json()
            if 'models' in result:
                model_names = [model.get('name', '') for model in result['models']]

                if self.model in model_names:
                    logger.info(f"Ollama is healthy. Model '{self.model}' is available.")
                    return True
                else:
                    logger.warning(
                        f"Ollama is running, but model '{self.model}' not found. "
                        f"Available models: {model_names}"
                    )
                    return False

            logger.info("Ollama is running.")
            return True

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            )
            return False

        except requests.exceptions.Timeout:
            logger.error("Health check timed out.")
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error during health check: {e}")
            return False

    def list_models(self) -> Optional[list]:
        """
        List all available models in Ollama.

        Returns:
            List of model names, or None if error occurs
        """
        try:
            response = requests.get(
                self.tags_url,
                timeout=5
            )
            response.raise_for_status()

            result = response.json()
            if 'models' in result:
                models = [model.get('name', 'unknown') for model in result['models']]
                logger.info(f"Available models: {models}")
                return models

            return []

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return None

    def set_model(self, model: str) -> None:
        """
        Change the model used by the client.

        Args:
            model: Name of the new model to use
        """
        old_model = self.model
        self.model = model
        logger.info(f"Changed model from '{old_model}' to '{model}'")


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize client
        client = OllamaClient(model="gemma2:2b")

        # Check health
        if client.check_health():
            print("Ollama is running!")

            # List available models
            models = client.list_models()
            if models:
                print(f"Available models: {models}")

            # Generate text
            prompt = "What is machine learning in one sentence?"
            response = client.generate(prompt, max_tokens=100)

            if response:
                print(f"\nPrompt: {prompt}")
                print(f"Response: {response}")
            else:
                print("Failed to generate response")
        else:
            print("Ollama is not running or model not available")

    except Exception as e:
        logger.error(f"Error in main: {e}")
