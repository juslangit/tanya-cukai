"""Tanya Cukai web server: serves the chat page and streams answers.

POST /api/ask {"question": "..."} returns newline-delimited JSON events:
    {"type": "sources", "sources": [...]}   the extracts the answer is built from
    {"type": "token", "text": "..."}        answer text as the model writes it
    {"type": "done"}
"""
import json
import pathlib
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app import rag  # noqa: E402
from app.embed import EMBED_URL  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent / "static"
PORT = 8000
index = None


def model_up(url):
    try:
        with urllib.request.urlopen(url.replace("/v1/chat/completions", "/health").replace("/v1/embeddings", "/health"), timeout=2):
            return True
    except Exception:
        return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self.send_json({"chat": model_up(rag.CHAT_URL), "embed": model_up(EMBED_URL),
                                   "chunks": len(index.meta)})
        name = "index.html" if self.path in ("/", "/index.html") else None
        if not name:
            return self.send_json({"error": "not found"}, 404)
        body = (STATIC / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/ask":
            return self.send_json({"error": "not found"}, 404)
        # only answer requests from this page, so another website open in the browser can't use the server
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"):
            return self.send_json({"error": "forbidden origin"}, 403)
        length = int(self.headers.get("Content-Length", 0))
        question = (json.loads(self.rfile.read(length) or b"{}").get("question") or "").strip()[:1000]
        if not question:
            return self.send_json({"error": "empty question"}, 400)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()

        try:
            hits = index.search(question)
            emit({"type": "sources", "sources": [
                {k: h[k] for k in ("title", "url", "location", "text", "score")} for h in hits]})
            for piece in rag.stream_answer(rag.build_messages(question, hits)):
                emit({"type": "token", "text": piece})
            emit({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # model server down, etc. — tell the page instead of hanging
            emit({"type": "error", "message": str(e)})


def main():
    global index
    index = rag.Index()
    print(f"Tanya Cukai: {len(index.meta)} chunks loaded — http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
