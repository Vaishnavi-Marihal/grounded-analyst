# Docs evaluation: changes to the gold list after seeing results

1. Coverage-after-flag question: allowed 0.77 and 0.74 (Qwen correctly said coverage rose "from 0.77 to 0.83"; my list had omitted the before-values).
2. "Share of true values inside the 90% range" question replaced with the rolling-backtest-folds question, because it could be answered from the API metrics and did not test the README search.
3. Deployment question: allowed 26.04 ("Ubuntu 26.04" appears in the README; my list had omitted it).

All three widen or clarify the test; none were made to turn a wrong answer into a pass. Results before each change are kept in results/.

# Held-out run (11 questions, setup frozen): findings about the checker

- Automatic result 7/11. Two of the four failures were substantively correct answers: one added a derived number (168 hours as "about 7 days"), one wrote 0.80 as 80%. The grounding rule treats numbers that are not in the retrieved text as invented, and does not convert units or percentages.
- These limits were NOT fixed after seeing the results. A later version should normalise percentages and allow simple unit conversions, and be tested on new questions.
- The other two failures were real: the answer omitted the requested category (Education), and described the test setup instead of the caveats.

# Manual review (15 random answers, qwen2.5:7b, 70-question set)

Reviewed by the author for readability only (numbers were not re-checked by hand).
All 15 read clearly. Automatic and manual judgement agreed on 14/15. The one
disagreement was a fluent answer with numbers fetched for the wrong date, which
a human reading the text cannot detect; only the independently computed gold
number catches it. 15 answers is a small sample and the reviewer is the author.
