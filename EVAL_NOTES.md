# Docs evaluation: changes to the gold list after seeing results

1. Coverage-after-flag question: allowed 0.77 and 0.74 (Qwen correctly said coverage rose "from 0.77 to 0.83"; my list had omitted the before-values).
2. "Share of true values inside the 90% range" question replaced with the rolling-backtest-folds question, because it could be answered from the API metrics and did not test the README search.
3. Deployment question: allowed 26.04 ("Ubuntu 26.04" appears in the README; my list had omitted it).

All three widen or clarify the test; none were made to turn a wrong answer into a pass. Results before each change are kept in results/.
