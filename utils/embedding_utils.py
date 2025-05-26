import numpy as np
from config.settings import get_openai_client, load_config


def get_embedding(text, model=None):
    config = load_config()
    if not model:
        model = config.get('embedding', {}).get('model', 'text-embedding-3-small')
    client = get_openai_client()
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
