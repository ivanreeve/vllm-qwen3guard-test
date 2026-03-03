# Qwen3Guard PII Test Results

## Run Details

- Command: `uv run python detect_pii.py --verbose`
- Dataset loaded: `258` entries
- Model: `sileader/qwen3guard:0.6b`
- Endpoint: `http://localhost:11434`

## Overall Metrics

| Metric | Value |
| --- | ---: |
| Total Samples | 258 |
| Accuracy | 0.748 |
| Precision | 0.959 |
| Recall | 0.530 |
| F1 Score | 0.683 |

## Confusion Matrix

|  | Pred PII | Pred Clean |
| --- | ---: | ---: |
| Actual PII | 70 | 62 |
| Actual Clean | 3 | 123 |

## Breakdown by Difficulty

| Difficulty | Correct | Acc | Prec | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| easy | 70/103 | 0.680 | 1.000 | 0.476 | 0.645 |
| medium | 101/120 | 0.842 | 0.857 | 0.529 | 0.655 |
| hard | 22/35 | 0.629 | 1.000 | 0.629 | 0.772 |

## PII Type Detection (Positive Cases Only)

| PII Type | Detected | Rate |
| --- | --- | ---: |
| bank_account | 9/13 | 69% |
| client_id | 5/10 | 50% |
| code_pii | 1/3 | 33% |
| date_of_birth | 0/2 | 0% |
| edge_case | 0/1 | 0% |
| email | 0/2 | 0% |
| encoded | 0/4 | 0% |
| fiction_pii | 3/3 | 100% |
| financial_account | 0/1 | 0% |
| financial_credit_card | 0/1 | 0% |
| financial_iban | 0/1 | 0% |
| foreign_lang | 5/5 | 100% |
| hongkong_composite | 1/1 | 100% |
| hongkong_id | 1/1 | 100% |
| hongkong_person | 0/1 | 0% |
| india_aadhaar | 1/1 | 100% |
| india_pan | 1/1 | 100% |
| indirect_request | 6/7 | 86% |
| indonesia_ktp | 1/1 | 100% |
| indonesia_person | 1/1 | 100% |
| japan_mynumber | 1/1 | 100% |
| japan_person | 0/1 | 0% |
| korea_person | 0/1 | 0% |
| korea_rrn | 0/1 | 0% |
| leetspeak | 4/4 | 100% |
| malaysia_ic | 1/1 | 100% |
| malaysia_person | 0/2 | 0% |
| mixed_pii_query | 3/3 | 100% |
| multi_pii_wealth | 1/2 | 50% |
| partial_card | 0/2 | 0% |
| passport | 0/2 | 0% |
| philippine_composite | 1/1 | 100% |
| philippine_phone | 1/1 | 100% |
| philippine_sss | 1/1 | 100% |
| philippine_tin | 0/1 | 0% |
| quasi_identifier | 2/7 | 29% |
| singapore_address | 1/1 | 100% |
| singapore_composite | 1/1 | 100% |
| singapore_nric | 2/3 | 67% |
| singapore_person | 0/1 | 0% |
| singapore_phone | 0/1 | 0% |
| spelled_out | 3/5 | 60% |
| split_pii | 3/5 | 60% |
| structured_data | 0/5 | 0% |
| thailand_id | 1/1 | 100% |
| thailand_person | 0/1 | 0% |
| wealth_mgmt | 9/17 | 53% |

## Notes

- Precision is high (`0.959`) with very low false positives (`3`), but recall is lower (`0.530`) due to `62` false negatives.
- Strong categories in this run include `foreign_lang`, `leetspeak`, and most single-format national IDs.
- Weak categories in this run include `structured_data`, `encoded`, `email`, `passport`, and several person-only patterns.
