"""Talk to the local bge-m3 embedding server (llama-server --embedding, port 8081)."""
import json
import os
import urllib.request

import numpy as np

EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")


def embed(texts):
    """Return one unit-length vector per text, so a dot product is the cosine similarity."""
    body = json.dumps({"input": texts}).encode()
    req = urllib.request.Request(EMBED_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)["data"]
    vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
