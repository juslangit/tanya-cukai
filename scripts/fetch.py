"""Download every LHDN source listed in sources.json into data/raw/.

The documents are public but belong to LHDN, so they are fetched on each machine
rather than committed to the repo.
"""
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/130 Safari/537.36"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for src in json.loads((ROOT / "sources.json").read_text()):
        ext = ".pdf" if src["url"].endswith(".pdf") else ".html"
        out = RAW / (src["id"] + ext)
        if out.exists():
            print("have ", out.name)
            continue
        req = urllib.request.Request(src["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            out.write_bytes(r.read())
        print("saved", out.name, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
