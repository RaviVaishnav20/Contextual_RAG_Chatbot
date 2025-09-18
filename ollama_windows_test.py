import requests

# Replace with your Ollama server URL
OLLAMA_GENERATE_URL = "http://192.168.1.138:11434/api/generate"

# Model and prompt
model_name = "gemma3:latest"
prompt = "Hello from Python test script!"

def test_ollama():
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        # Ollama returns 'response' field directly
        text = response.json().get("response", "")
        print("Ollama response:\n", text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_ollama()
