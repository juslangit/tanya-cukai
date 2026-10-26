"""Turn the downloaded LHDN sources into a searchable index.

    raw page/PDF  ->  sections (heading + text)  ->  chunks  ->  embeddings  ->  data/index.sqlite

Embeddings come from the local bge-m3 model served by llama-server (see run.sh),
so nothing leaves the machine.
"""
import html.parser
import json
import re
import pathlib
import sqlite3
import sys

import numpy as np
from pypdf import PdfReader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.embed import embed  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "index.sqlite"

CHUNK_CHARS = 900   # about 200-250 tokens: small enough to be specific, big enough to keep a rule whole
OVERLAP = 150       # repeat the tail of a chunk at the start of the next, so a sentence cut in half is still found
OLDEST_YEAR = 2024  # LHDN keeps relief and rate tables back to 2013; older years only crowd out the current ones


class PageText(html.parser.HTMLParser):
    """Collect the text inside the page's main Elementor block, split at headings.

    LHDN pages are WordPress/Elementor: the header and footer are their own blocks,
    and the article sits in the block marked data-elementor-type="wp-page".
    """

    HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    SKIP = {"script", "style", "svg", "noscript"}

    def __init__(self):
        super().__init__()
        self.depth = 0          # div depth inside the wp-page block; 0 = outside it
        self.skip = 0
        self.in_heading = False
        self.in_cell = 0
        self.sections = [{"heading": "", "text": []}]

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if self.depth == 0:
            if tag == "div" and a.get("data-elementor-type") == "wp-page":
                self.depth = 1
            return
        if tag == "div":
            self.depth += 1
        if tag in self.SKIP:
            self.skip += 1
        # accordion titles act as headings (the relief list is one big accordion)
        cls = a.get("class") or ""
        if tag in self.HEADINGS or "accordion-item-title-text" in cls:
            self.in_heading = True
            self.sections.append({"heading": "", "text": []})
        if tag in ("td", "th"):
            self.in_cell += 1
            self.sections[-1]["text"].append(" | ")
        elif tag == "tr" or (tag in ("p", "li", "br") and not self.in_heading):
            # inside a table cell a new paragraph is just a space, so each row stays on one line
            self.sections[-1]["text"].append(" " if self.in_cell and tag != "tr" else "\n")

    def handle_endtag(self, tag):
        if self.depth == 0:
            return
        if tag == "div":
            self.depth -= 1
        if tag in self.SKIP:
            self.skip -= 1
        if tag in ("td", "th"):
            self.in_cell -= 1
        if tag in self.HEADINGS or (self.in_heading and tag in ("div", "span")):
            self.in_heading = False

    def handle_data(self, data):
        if self.depth == 0 or self.skip:
            return
        # a newline in the HTML source is just a space to the browser; real line breaks come from tags
        data = data.replace("\n", " ")
        if self.in_heading:
            self.sections[-1]["heading"] += data.strip() + " "
        else:
            self.sections[-1]["text"].append(data)


def clean(text):
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line.strip(" |"))


def sections_from_html(path):
    p = PageText()
    p.feed(path.read_text(encoding="utf-8", errors="ignore"))
    out = []
    for s in p.sections:
        text = clean("".join(s["text"]))
        # skip the side menu repeated on every page, and the bare "Expand All" toggles
        if text and not text.startswith("Introduction Individual Income Tax") and text != "Expand All":
            out.append({"where": " ".join(s["heading"].split()), "text": text})
    return out


def sections_from_pdf(path):
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = clean(page.extract_text() or "")
        if text:
            out.append({"where": f"page {i}", "text": text})
    return out


def chunk(text):
    """Split on line boundaries into ~CHUNK_CHARS pieces with a little overlap.

    A table (most lines start with "|") is split one row per piece instead, each carrying the
    header row, so a single relief or tax band is found on its own rather than buried in a
    block of unrelated rows."""
    lines = text.split("\n")
    table = [i for i, line in enumerate(lines) if line.startswith("|")]
    if len(table) > 2 and len(table) / len(lines) >= 0.6:
        # text above the table is chunked normally; the first table line is the column header
        intro = "\n".join(lines[:table[0]])
        header = lines[table[0]]
        rows = [line for line in lines[table[0] + 1:] if line.strip()]
        return (chunk(intro) if intro else []) + [header + "\n" + row for row in rows]
    if len(text) <= CHUNK_CHARS:
        return [text]
    pieces, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) > CHUNK_CHARS:
            pieces.append(cur)
            cur = cur[-OVERLAP:]
        cur = (cur + "\n" + line).strip()
    if cur:
        pieces.append(cur)
    return pieces


def is_old_year(where):
    """True for a 'Year of Assessment ...' section whose newest year is before OLDEST_YEAR."""
    if "Year of Assessment" not in where:
        return False
    years = [int(y) for y in re.findall(r"20\d\d", where)]
    return bool(years) and max(years) < OLDEST_YEAR


def main():
    sources = {s["id"]: s for s in json.loads((ROOT / "sources.json").read_text())}
    rows = []
    for sid, src in sources.items():
        path = next(RAW.glob(sid + ".*"))
        secs = sections_from_pdf(path) if path.suffix == ".pdf" else sections_from_html(path)
        secs = [s for s in secs if not is_old_year(s["where"])]
        for s in secs:
            for piece in chunk(s["text"]):
                rows.append((sid, src["title"], src["url"], s["where"], piece))
        print(f"{sid:24} {len(secs):4} sections")

    print(f"embedding {len(rows)} chunks ...")
    # the heading goes into the embedded text too, so "relief for parents" finds the parents section
    texts = [f"{title} — {where}\n{text}" for _, title, _, where, text in rows]
    vecs = []
    for i in range(0, len(texts), 16):
        vecs.extend(embed(texts[i:i + 16]))
        print(f"  {min(i + 16, len(texts))}/{len(texts)}", end="\r")
    print()

    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.execute("create table chunks (id integer primary key, source text, title text, url text,"
                " location text, text text, vec blob)")
    con.executemany(
        "insert into chunks (source, title, url, location, text, vec) values (?,?,?,?,?,?)",
        [(*r, np.asarray(v, dtype=np.float32).tobytes()) for r, v in zip(rows, vecs)],
    )
    con.commit()
    print("saved", DB, len(rows), "chunks")


if __name__ == "__main__":
    main()
