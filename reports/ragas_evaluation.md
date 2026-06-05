# RAGAS Generation Evaluation and Final Selection

The table below compares all retrieval methods after answer generation. The final method is selected only by RAGAS generation metrics.

- Backend: `ragas_evaluate`
- Cached metrics used: `True`
- Answers cached: `True`
- Answers evaluated: `180`
- Retrieval methods evaluated: `BM25, Vector, Hybrid, Hybrid + RRF, Hybrid + RRF + Cross-Encoder, Hybrid + RRF + Cross-Encoder + MMR`
- Selected final method: `Hybrid + RRF + Cross-Encoder + MMR`
- Selection metric: `generation_score`
- Selection formula: `generation_score = 0.40*faithfulness + 0.30*answer_relevancy + 0.30*answer_correctness`
- Retrieval metrics used for selection: `False`
- Generation score note: `Method-level metric means ignore RAGAS NaN rows; metric_valid_counts reports coverage.`
- RAGAS context max chars per chunk: `1200`

| Rank | Method | Retrieval score | Faithfulness | Answer relevancy | Answer correctness | Context precision | Context recall | Generation score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Hybrid + RRF + Cross-Encoder + MMR | 0.894 | 0.870 | 0.519 | 0.474 | N/A | N/A | 0.646 |
| 2 | Hybrid + RRF | 0.890 | 0.771 | 0.582 | 0.500 | N/A | N/A | 0.633 |
| 3 | BM25 | 0.889 | 0.875 | 0.533 | 0.323 | N/A | N/A | 0.607 |
| 4 | Hybrid | 0.893 | 0.713 | 0.553 | 0.434 | N/A | N/A | 0.582 |
| 5 | Vector | 0.882 | 0.889 | 0.334 | 0.353 | N/A | N/A | 0.562 |
| 6 | Hybrid + RRF + Cross-Encoder | 0.894 | 0.714 | 0.490 | 0.344 | N/A | N/A | 0.536 |

## Metric Coverage

| Method | Faithfulness n | Answer relevancy n | Answer correctness n |
|---|---:|---:|---:|
| Hybrid + RRF + Cross-Encoder + MMR | 9 | 19 | 9 |
| Hybrid + RRF | 6 | 18 | 11 |
| BM25 | 4 | 11 | 5 |
| Hybrid | 9 | 7 | 11 |
| Vector | 9 | 15 | 9 |
| Hybrid + RRF + Cross-Encoder | 7 | 14 | 7 |

## Unavailable Metrics

- None

## Figures

- `reports/figures/generation_score.png`
- `reports/figures/generation_metric_breakdown.png`
- `reports/figures/generation_query_heatmap.png`

## Weakest Cases

- `Vector` - CHƯƠNG VI nói về cái gì vậy ạ? - mean=0.000
- `Vector` - Theo chỉ thị của UBNND TỈNH LÂM ĐỒNG, tình hình thực trạng quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh hiện nay đang gặp phải những vấn đề gì? - mean=0.000
- `Hybrid` - Theo chỉ thị của UBNND TỈNH LÂM ĐỒNG, tình hình thực trạng quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh hiện nay đang gặp phải những vấn đề gì? - mean=0.000
- `Vector` - Vai trò của Cục quản lý vốn & TSNN trong đào tạo nghề tại doanh nghiệp là gì? - mean=0.000
- `Hybrid + RRF` - Theo chỉ thị của UBNND TỈNH LÂM ĐỒNG, tình hình thực trạng quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh hiện nay đang gặp phải những vấn đề gì? - mean=0.000
- `Hybrid + RRF` - Cty em phai chi bao nhieu % quy luong cho LD-TB & XH de dao tao nghe? - mean=0.000
- `Hybrid + RRF` - Theo chỉ thị của UBND tỉnh Lâm Đồng, tình hình quản lý hoạt động dạy nghề trên địa bàn tỉnh hiện nay đang gặp phải những hạn chế gì? - mean=0.000
- `Hybrid + RRF` - Khi nào thì hội đồng quản trị của xí nghiệp liên doanh phải họp bàn và quyết định việc giải thể theo quy định, và họ cần làm gì khi giải thể ạ? - mean=0.000
- `Hybrid + RRF + Cross-Encoder + MMR` - Quy trình giải thể Xí nghiệp liên doanh được quy định như thế nào khi gặp thua lỗ? - mean=0.000
- `Hybrid + RRF + Cross-Encoder + MMR` - Khi nào thì hội đồng quản trị của xí nghiệp liên doanh phải họp bàn và quyết định việc giải thể theo quy định, và họ cần làm gì khi giải thể ạ? - mean=0.000
