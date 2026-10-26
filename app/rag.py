"""Retrieval and answering: find the LHDN extracts that match a question, then have the
local model answer from those extracts only, citing them by number."""
import json
import os
import pathlib
import sqlite3
import urllib.request

import numpy as np

from app.embed import embed

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "index.sqlite"
CHAT_URL = os.environ.get("CHAT_URL", "http://127.0.0.1:8080/v1/chat/completions")
TOP_K = 6

SYSTEM = """You are Tanya Cukai, an assistant for Malaysian individual income tax questions.

Rules:
- Answer ONLY from the numbered LHDN extracts given to you. Do not use outside knowledge.
- Start with the direct answer in one sentence, including the amount or date. Then add at most
  three short points on the conditions or limits that matter. No long conclusions.
- Cite the extract you used after each fact, like [1] or [2][4].
- Reply in the same language and style as the question. For Malay, write Malaysian Malay
  (Bahasa Malaysia), never Indonesian: "rawatan perubatan" not "perawatan medis",
  "datuk nenek" not "kakek nenek". Rojak questions get a casual rojak reply.
- Keep separate things separate: a penalty for filing late is not the same as a penalty for paying late.
- Say which Year of Assessment (Tahun Taksiran) a figure applies to when the extract says it.
- If the extracts answer only part of the question, answer that part and say what is missing.
  If they do not answer it at all, say so and suggest checking hasil.gov.my. Never guess an amount.
- You give general information, not personal tax advice."""


class Index:
    """All chunk vectors held in one numpy matrix: a few thousand rows fit easily in memory,
    and a matrix product scores every chunk at once."""

    def __init__(self, path=DB):
        con = sqlite3.connect(path)
        rows = con.execute("select id, source, title, url, location, text, vec from chunks").fetchall()
        self.meta = [dict(id=r[0], source=r[1], title=r[2], url=r[3], location=r[4], text=r[5]) for r in rows]
        self.vecs = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])

    def search(self, question, k=TOP_K):
        q = embed([question])[0]
        scores = self.vecs @ q
        best = np.argsort(-scores)[:k]
        return [dict(self.meta[i], score=float(scores[i])) for i in best]


def build_messages(question, hits):
    extracts = "\n\n".join(
        f"[{n}] {h['title']} — {h['location']}\n{h['text']}" for n, h in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"LHDN extracts:\n\n{extracts}\n\nQuestion: {question}"},
    ]


def stream_answer(messages):
    """Yield the answer text piece by piece from llama-server's OpenAI-style streaming API."""
    body = json.dumps({"messages": messages, "stream": True, "temperature": 0.2, "max_tokens": 700}).encode()
    req = urllib.request.Request(CHAT_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
            if delta:
                yield delta
