# Phase 3 results

Model: `google_genai:gemini-3.5-flash` · semantic backend: `hf` · 107 pages, 516 chunks · generated 2026-09-30 19:18:43 +0000

## Retrieval (Q1–Q6)

| Retriever | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | string hits | page hits |
|---|---|---|---|---|---|---|---|---|
| semantic@4 | HIT [36, 89, 31, 32] | miss [4, 86, 6, 36] | HIT [37, 38, 47, 31] | HIT [36, 36, 31, 43] | miss [46, 46, 82, 88] | HIT [39, 39, 39, 32] | 4/6 | 5/6 |
| keyword@4 | miss [42, 43, 41, 40] | HIT [28, 9, 79, 9] | HIT [37, 38, 48, 47] | HIT [36, 31, 41, 40] | HIT [14, 41, 14, 32] | miss [32, 39, 37, 39] | 4/6 | 5/6 |
| semantic@6 | HIT [36, 89, 31, 32, 4, 35] | miss [4, 86, 6, 36, 89, 9] | HIT [37, 38, 47, 31, 38, 47] | HIT [36, 36, 31, 43, 37, 89] | miss [46, 46, 82, 88, 45, 46] | HIT [39, 39, 39, 32, 31, 36] | 4/6 | 6/6 |
| keyword@6 | miss [42, 43, 41, 40, 6, 44] | HIT [28, 9, 79, 9, 78, 96] | HIT [37, 38, 48, 47, 47, 38] | HIT [36, 31, 41, 40, 5, 6] | HIT [14, 41, 14, 32, 13, 45] | miss [32, 39, 37, 39, 87, 31] | 4/6 | 5/6 |
| hybrid@4 | HIT [36, 42, 89, 43] | HIT [4, 28, 86, 9] | HIT [37, 38, 47, 48] | HIT [36, 31, 36, 41] | HIT [46, 14, 46, 41] | HIT [32, 39, 39, 39] | 6/6 | 6/6 |
| hybrid@6 | HIT [36, 42, 89, 43, 31, 41] | HIT [4, 28, 86, 9, 6, 79] | HIT [37, 38, 47, 48, 31, 47] | HIT [36, 31, 36, 41, 43, 40] | HIT [46, 14, 46, 41, 82, 14] | HIT [32, 39, 39, 39, 37] | 6/6 | 6/6 |

## Answers — baseline (Phase 2: semantic@4)

| Q | answer | answerable | cited pages | answer_ok | cited_ok |
|---|---|---|---|---|---|
| Q1 | In fiscal 2023, NIKE, Inc.'s total revenues were $51.2 billion (or $51,217 million). | True | [36, 89, 31, 32] | True | True |
| Q2 | I am sorry, but the provided excerpts do not contain information about the number of employees NIKE had as of May 31, 2023. | False | [] | False | False |
| Q3 | In fiscal 2023, consolidated gross margin decreased 250 basis points to 43.5% compared to 46.0% for fiscal 2022. Meanwhile, consolidated gross profit increased 4% to $22,292 million compared to $21,479 million for fiscal 2022. | True | [37] | True | True |
| Q4 | NIKE Direct revenues grew 14% from $18.7 billion in fiscal 2022 to $21.3 billion in fiscal 2023. On a currency-neutral basis, NIKE Direct revenues increased 20%. | True | [36, 31] | True | True |
| Q5 | NIKE conducts business in various currencies, which subjects it to foreign currency risk that can impact its consolidated results of operations, financial position, and cash flows. The company's foreign currency exposures include:  1. **Product Costs:** Primarily generated through non-functional currency denominated product purchases. Specifically, a weakening U.S. Dollar against foreign currencies within the factory currency exposure indices increases NIKE's U.S. Dollar inventory cost, while a strengthening U.S. Dollar reduces it. 2. **Non-functional currency denominated revenues** 3. **Intercompany revenues** 4. **Demand creation expenses** 5. **Investments in U.S. Dollar denominated available-for-sale debt securities** 6. **Certain other intercompany transactions** | True | [46, 46, 82] | None | True |
| Q6 | To find the EBIT margin for each NIKE Brand geographic segment in fiscal 2023, we divide each segment's EBIT by its Revenues:  1. **Greater China**:    - EBIT: $2,283 million [2]    - Revenues: $7,248 million [1]    - EBIT Margin: $2,283 / $7,248 ≈ 31.5%  2. **Asia Pacific & Latin America**:    - EBIT: $1,932 million [3]    - Revenues: $6,431 million [1]    - EBIT Margin: $1,932 / $6,431 ≈ 30.0%  3. **Europe, Middle East & Africa**:    - EBIT: $3,531 million [2]    - Revenues: $13,418 million [1]    - EBIT Margin: $3,531 / $13,418 ≈ 26.3%  4. **North America**:    - EBIT: $5,454 million [2]    - Revenues: $21,608 million [1]    - EBIT Margin: $5,454 / $21,608 ≈ 25.2%  Therefore, **Greater China** had the highest EBIT margin in fiscal 2023 at approximately 31.5%. | True | [39, 39, 39, 32] | True | True |
| Q7 | I am sorry, but the provided excerpts do not contain information about NIKE's total revenues in fiscal 2025. The data only covers up to fiscal 2023. | False | [] | True | None |

## Answers — final (Phase 3: hybrid@6)

| Q | answer | answerable | cited pages | answer_ok | cited_ok |
|---|---|---|---|---|---|
| Q1 | In fiscal 2023, NIKE, Inc.'s total revenues were $51.2 billion (or $51,217 million). | True | [36, 89, 31] | True | True |
| Q2 | As of May 31, 2023, NIKE had approximately 83,700 employees worldwide, including retail and part-time employees. | True | [9] | True | True |
| Q3 | In fiscal 2023, gross margin decreased 250 basis points to 43.5% compared to 46.0% for fiscal 2022. This decrease was primarily driven by:  - Higher NIKE Brand product costs, on a wholesale equivalent basis, primarily due to higher input costs and elevated inbound freight and logistics costs as well as product mix; - Lower margin in the NIKE Direct business, driven by higher promotional activity to liquidate inventory in the current period compared to lower promotional activity in the prior period;  - Unfavorable changes in net foreign currency exchange rates, including hedges; and - Lower off-price margin, on a wholesale equivalent basis. | True | [37] | True | True |
| Q4 | In fiscal 2023, NIKE Direct revenues increased 14% (or 20% on a currency-neutral basis) from $18.7 billion in fiscal 2022 to $21.3 billion. | True | [36, 31] | True | True |
| Q5 | According to the provided filings, NIKE faces several risks from fluctuations in foreign currency exchange rates:  1. **Financial Impact:** Fluctuations could result in lower revenues, higher costs, decreased margins and earnings, and have an adverse effect on results of operations and financial condition [2, 6].  2. **Translation Effects:** Because international revenues and expenses are derived from sales and operations in foreign currencies, a weakening of foreign currencies relative to the U.S. Dollar adversely affects the U.S. Dollar value of the Company's foreign currency-denominated sales and earnings [6]. 3. **Product and Inventory Costs:** NIKE faces transactional foreign currency risks, primarily related to product costs from non-functional currency denominated product purchases [3, 5]. For example, a weakening U.S. Dollar against foreign currencies in the factory currency exposure index increases NIKE's inventory cost [1]. 4. **Supply Chain Disruption:** Currency exchange rate fluctuations can disrupt the business of independent manufacturers that produce NIKE's products by making their purchases of raw materials more expensive and more difficult to finance [6]. | True | [46, 14, 46, 82, 14] | None | True |
| Q6 | Based on the provided excerpts, the EBIT margins for the NIKE Brand geographic segments in fiscal 2023 are calculated as follows:  * **Greater China**: EBIT of $2,283 million divided by Revenues of $7,248 million equals approximately **31.50%**. * **Asia Pacific & Latin America**: EBIT of $1,932 million divided by Revenues of $6,431 million equals approximately **30.04%**. * **Europe, Middle East & Africa**: EBIT of $3,531 million divided by Revenues of $13,418 million equals approximately **26.31%**. * **North America**: EBIT of $5,454 million divided by Revenues of $21,608 million equals approximately **25.24%**.  Therefore, **Greater China** had the highest EBIT margin among the NIKE Brand geographic segments in fiscal 2023. | True | [39, 39, 39] | True | True |
| Q7 | I am sorry, but the provided excerpts do not contain information about NIKE's total revenues in fiscal 2025, as the data only goes up to fiscal 2023. | False | [] | True | None |

## Multi-turn follow-up (expected: 16% currency-neutral)

- **without question rewriting (Phase 2)** — retrieval query: “How much did that grow on a currency-neutral basis?” → On a currency-neutral basis, NIKE, Inc. Revenues increased 16% in fiscal 2023 compared to fiscal 2022. (answer_ok=True, history messages=4)
- **with question rewriting (Phase 3)** — retrieval query: “How much did NIKE, Inc.'s total revenues grow on a currency-neutral basis in fiscal 2023?” → In fiscal 2023, NIKE, Inc.'s revenues increased 16% on a currency-neutral basis compared to fiscal 2022. (answer_ok=True, history messages=4)
