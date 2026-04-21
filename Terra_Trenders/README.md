## Approach Report:

We evaluated one LLM endpoint after initially testing three endpoints; the two mini-LLM endpoints failed because of input/output token limitations. We compared multiple prompt versions on the same held-out subset from Week 3. To address the token limit issue, we attempted to shorten the prompt input by reducing the number of characters, but this did not address the issue with the two failed endpoints. We then selected the best-performing prompt from four candidates, based on two prompt versions and whether to include a few-shot example. Selection was based on parse reliability and accuracy. The outputs for event counts across the 16 classes were strictly constrained to JSON format and parsed before scoring.

## Error Analysis:

Common failures included malformed JSON, invalid value types or ranges, and inconsistent formatting across models. Prompt wording slightly affected compliance and output quality, but its overall impact is difficult to assess because of the limited regression scope, including cases where no event was predicted. Few-shot prompting often improved reliability and, in some cases, determined whether regression can be possible when testing the three LLM endpoints.

## Comparison Summary:

Based on MAE by region, the statistical regression model performed better overall than the LLM-based regression in predicting the number of events for each class. This is consistent with the fundamental difference between the two approaches: the LLM relies on contextual matching, while the statistical model is designed specifically for regression. However, there is also a key distinction in output format: the LLM predicts exact integer counts, whereas the statistical regression produces estimated counts as floating-point values.

| Region | LLM MAE | Regression MAE |
|---|---:|---:|
| NERC1  | 4.833333 | 2.397490 |
| NERC2  | 6.166667 | 1.170530 |
| NERC3  | 4.833333 | 3.383684 |
| NERC4  | 5.500000 | 2.022420 |
| NERC5  | 3.833333 | 2.626505 |
| NERC6  | 3.833333 | 2.373134 |
| NERC7  | 2.666667 | 2.919246 |
| NERC8  | 5.833333 | 3.916989 |
| NERC9  | 6.166667 | 2.667303 |
| NERC10 | 6.666667 | 1.877288 |
| NERC11 | 6.000000 | 2.838670 |
| NERC12 | 3.166667 | 2.973450 |
| NERC15 | 4.166667 | 2.827884 |
| NERC17 | 2.833333 | 3.441790 |
| NERC18 | 4.833333 | 2.934488 |
| NERC20 | 3.500000 | 3.010190 |

## Repeatability Steps:

Run the following notebook: `/Notebooks/llm_prediction.ipynb`
