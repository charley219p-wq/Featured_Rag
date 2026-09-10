from typing import List
from langchain_core.embeddings import Embeddings
import requests
import os

class GenerativeEngineEmbeddings(Embeddings):

    def __init__(
        self,
        api_key=None,
        model="amazon.titan-embed-text-v2:0",
        base_url="https://openai.generative.engine.capgemini.com/v1"
    ):

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.embeddings_url = f"{base_url}/embeddings"

    def _get_embedding(self, text):

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            "model": self.model
        }

        response = requests.post(
            self.embeddings_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:
            embeddings.append(
                self._get_embedding(text)
            )

        return embeddings

    def embed_query(self, text):

        return self._get_embedding(text)