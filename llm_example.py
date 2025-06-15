import requests
import json

def query_ollama(prompt: str, model: str = "gemma3:1b") -> str:
    """
    Send a prompt to Ollama and get the response
    """
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "seed": 0,
        "top_k": 10,
        "top_p": 0.95,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    return response.json()["response"]

def demonstrate_llm_inconsistency():

    base = "GOAL: Be concise and clear - INPUT: "
    
    # Slightly modified prompts that ask for the same core information but with different framing.
    prompts = [
        base + "Describe a house cat in a single sentence.",
        base + "Describe a house cat in a single sentence.",
        base + " Describe a house cat in a single setence. ",
        base + "In one-sentence describe of a house cat.",
        base + "describe a house cat in a short sentence."
    ]
        
    # Try each prompt and show the differences
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}: '{prompt}'")
        response = query_ollama(prompt)
        print(f"Response {i}: '{response}'\n")

if __name__ == "__main__":
    print("Starting LLM inconsistency demonstration...")
    demonstrate_llm_inconsistency()