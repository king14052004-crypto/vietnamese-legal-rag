# RAGAS Answer Evaluation and Final Selection

## Comparison

The table below compares the finalist retrieval methods after generation. The final method is chosen only after looking at retrieval score, answer score, and the combined final score.

- Backend: `ragas_evaluate`
- Cached metrics used: `True`
- Answers cached: `True`
- Answers evaluated: `60`
- Retrieval methods evaluated: `hybrid, bm25`
- Selected final method: `hybrid`
- Selection formula: `final_score = 0.45 * retrieval_candidate_score + 0.55 * answer_score`

| Rank | Method | Retrieval score | Answer score | Final score |
|---:|---|---:|---:|---:|
| 1 | hybrid | 0.533 | 0.576 | 0.556 |
| 2 | bm25 | 0.532 | 0.000 | 0.239 |

## RAGAS Metrics

| Method | Metric | Score |
|---|---|---:|
| bm25 | context_precision | 0.000 |
| hybrid | context_precision | 0.484 |
| hybrid | context_recall | 0.667 |
| all | answer_correctness | unavailable |
| all | answer_relevancy | unavailable |
| all | faithfulness | unavailable |

## Figures

- `reports/figures/final_method_selection.png`
- `reports/figures/final_score_breakdown.png`
- `reports/figures/ragas_per_question_heatmap.png`

## Weakest Cases

- `hybrid` - Theo quy định tại CHƯƠNG VI thì khi xí nghiệp liên doanh giải thể, hội đồng quản trị cần phải thực hiện những nghĩa vụ gì đối với người lao động và các bên liên quan? - mean=0.000
- `hybrid` - Quy trình giải thể Xí nghiệp liên doanh được quy định như thế nào khi gặp thua lỗ? - mean=0.000
- `hybrid` - Khi nào thì hội đồng quản trị của xí nghiệp liên doanh phải họp bàn và quyết định việc giải thể theo quy định, và họ cần làm gì khi giải thể ạ? - mean=0.000
- `hybrid` - CHƯƠNG VI nói về cái gì vậy ạ? - mean=0.000
- `hybrid` - Văn bản này hướng dẫn thực hiện các quy định về lao động dựa trên nghị định nào của Chính phủ? - mean=0.000
- `hybrid` - Theo chỉ thị của UBNND TỈNH LÂM ĐỒNG, tình hình thực trạng quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh hiện nay đang gặp phải những vấn đề gì? - mean=0.000
- `hybrid` - Theo chỉ thị của UBND tỉnh Lâm Đồng, tình hình quản lý hoạt động dạy nghề trên địa bàn tỉnh hiện nay đang gặp phải những hạn chế gì? - mean=0.000
- `hybrid` - UBND TỈNH LÂM ĐỒNG đã đánh giá như thế nào về tình hình quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh? - mean=0.000
