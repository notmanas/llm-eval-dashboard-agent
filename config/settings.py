import openai
from dotenv import load_dotenv
import os
import yaml

load_dotenv()


def load_config() -> dict:
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)


def get_openai_client() -> openai.OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    return openai.OpenAI(api_key=api_key)
