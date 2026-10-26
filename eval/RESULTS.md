# Evaluation results

Test set: `questions.json`, 24 questions (8 BM, 9 English, 7 rojak), each tied to the LHDN passage
holding the answer. Machine: MacBook Air M2, 8 GB. Models: Qwen3-4B-Instruct-2507 Q4_K_M (answers),
bge-m3 Q8_0 (embeddings), both under llama.cpp.

## Retrieval — is the answer passage in the top K? (2026-10-26)

One change at a time, re-measured after each.

| Step | Change | hit@1 | hit@3 | hit@6 | Kind |
|---|---|---|---|---|---|
| 0 | Baseline: ~900-char chunks, every year's tables | 8 | 17 | 21 | — |
| 1 | Test label for "pinda BE" was too narrow (the #2 result was a correct answer) | 8 | 18 | 22 | label fix |
| 2 | Each table row its own chunk, with the header row | 8 | 18 | 21 | system — mixed: fixed home-loan, but 1,634 near-identical rows from 2016–2023 crowded others out |
| 3 | Index only YA 2024–2025 from the year-by-year relief and rate tables | 8 | 20 | 21 | system |
| 4 | Two parser bugs: HTML-source newlines split table cells; a paragraph above a table was used as every row's header. Plus a label fix: FAQ 48 also states the payment code | 10 | 21 | 23 | system + label fix |
| 5 | Bilingual source titles (the pages are English, the titles were Malay-only) | 11 | 23 | 23 | system |
| 6 | Test matching looked at the passage body only, case-sensitively; the model also sees the heading | 12 | 24 | 24 | label fix |

**Final: hit@3 24/24, hit@1 12/24.** Three of the gains were corrections to my own test, not to
the system; with the test fixed from the start, the system changes took hit@3 from 18 to 24.
24 questions is small and I tuned against them, so the next step is a held-out set written
without looking at the results.

## Answers — graded by hand against the LHDN text (2026-10-26)

System prompt v2 (direct answer first, Malaysian Malay, keep filing and payment penalties apart).
Full answers in `answers.json`. First word in ~4 s, full answer 8–14 s.

| Result | Count | Questions |
|---|---|---|
| Correct headline answer | 20 / 24 | |
| Partly right | 2 | #18 RM400 rebate (said joint assessment only), #21 amending BE (missed the 6-month ARF route) |
| Wrong | 2 | #11 tax on RM50,000 (contradicts itself on 3/6/11%), #13 Form B deadline (invented the year 2025; should be 2026) |
| Correct, but with a wrong side point | 4 | #7, #16, #17, #19 (e.g. mixes the umrah departure levy into zakat) |
| Rojak question answered in rojak | 0 / 7 | replies in formal BM; Indonesian words still appear ("tergantung", "dikoreksi", "kisaran") |

Prompt v1 (before the rewrite) used Indonesian words ("perawatan medis", "kakek nenek"), hedged for a
paragraph before giving a figure it had found, and gave the 10% late *payment* penalty for late *filing*.

## What this points to

- **Arithmetic belongs in code.** #11 is a calculation; the model should get the band table and a
  computed figure, not do the sum itself.
- **Tone and register are a LoRA job** (phase 2): rojak replies to rojak questions, Malaysian rather
  than Indonesian Malay. Prompting fixed part of it and stalled.
- **Facts stay in retrieval.** Every wrong answer had the right passage in front of it; none came
  from a missing document.
