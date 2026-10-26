"""Measure retrieval: for each test question, is the passage holding the answer in the top K?

    uv run eval/run_eval.py            # retrieval only (fast)
    uv run eval/run_eval.py --answers  # also generate answers and save them for reading

Needs run.sh's model servers to be up.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import rag  # noqa: E402


def main():
    qs = json.loads((ROOT / "eval" / "questions.json").read_text())
    index = rag.Index()
    hits_at = {1: 0, 3: 0, 6: 0}
    by_lang = {}
    answers = []
    for item in qs:
        # some answers are stated on more than one LHDN page; any of them counts
        sources = item["source"] if isinstance(item["source"], list) else [item["source"]]
        found = index.search(item["q"], k=6)
        rank = next((i + 1 for i, h in enumerate(found)
                     if h["source"] in sources and item["must"].lower() in (h["location"] + "\n" + h["text"]).lower()), None)
        for k in hits_at:
            hits_at[k] += bool(rank and rank <= k)
        lang = by_lang.setdefault(item["lang"], [0, 0])
        lang[0] += bool(rank)
        lang[1] += 1
        print(f"{'ok ' if rank else 'MISS'} rank={rank or '-'}  {item['q']}")
        if "--answers" in sys.argv:
            text = "".join(rag.stream_answer(rag.build_messages(item["q"], found)))
            answers.append({**item, "rank": rank, "model_answer": text})

    n = len(qs)
    print("\n" + "  ".join(f"hit@{k}: {v}/{n} ({v / n:.0%})" for k, v in hits_at.items()))
    print("  ".join(f"{lang}: {a}/{b}" for lang, (a, b) in by_lang.items()))
    if answers:
        out = ROOT / "eval" / "answers.json"
        out.write_text(json.dumps(answers, ensure_ascii=False, indent=1))
        print("answers saved to", out)


if __name__ == "__main__":
    main()
