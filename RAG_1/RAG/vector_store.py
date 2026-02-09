from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Config
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def init_embed_model(model_name: str = EMBED_MODEL_NAME):
    print("Loading embedding model:", model_name)
    model = SentenceTransformer(model_name)
    print("Model loaded.")
    return model

# new_chroma_init.py
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from typing import Tuple

def init_chroma_client(persist_directory: str = "./chroma_db", collection_name: str = "docs") -> Tuple[object, object]:
    """
    Initialize a Chroma persistent client (new API). Returns (client, collection).
    If persistent client cannot be created, falls back to an in-memory client.
    """
    try:
        # ensure dir exists
        os.makedirs(persist_directory, exist_ok=True)

        # PersistentClient is the new recommended local client class
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(),                 # you may pass custom Settings() here
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )

        # get or create collection (same helper exists on new client)
        coll = client.get_or_create_collection(collection_name)
        print("Chroma PersistentClient initialized at:", persist_directory)
        return client, coll

    except Exception as e:
        # fallback: in-memory client for quick testing
        print("PersistentClient init failed (falling back to in-memory). Exception:", e)
        try:
            client = chromadb.Client()   # memory-only default on many versions
            coll = client.get_or_create_collection(collection_name)
            print("Using in-memory Chroma client.")
            return client, coll
        except Exception as e2:
            # last-resort: HttpClient (if a server is available)
            print("In-memory client also failed:", e2)
            raise RuntimeError("Unable to create any Chroma client; check chromadb version and installation.") from e2


# --- helper: embed texts (returns numpy array) ---
def embed_texts(model, texts):
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embs, dtype=np.float32)
