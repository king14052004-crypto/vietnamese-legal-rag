# RAGAS Answer Evaluation

- Backend: `ragas_evaluate`
- Cached artifact used: `True`
- Cache reason: `cached_answers`
- Answers: `30`
- Retrieval method: `hybrid`
- Model: `gemini-3.1-flash-lite`

| Metric | Score |
|---|---:|
| context_precision | 0.528 |
| context_recall | 0.773 |
| faithfulness | 0.876 |
| answer_relevancy | 0.760 |
| answer_correctness | 0.640 |

## Figures

- `reports/figures/ragas_answer_metrics.png`
- `reports/figures/ragas_per_question_heatmap.png`

## Weakest Cases

- Văn bản này hướng dẫn thực hiện các quy định về lao động dựa trên nghị định nào của Chính phủ? - mean=0.062
- Cho em hoi la cai Sở Lao động liên khu no co nhiem vu gi trong viec giai quyet mau thuan giua chu voi cong nhan khong a? - mean=0.246
- Theo quy định tại CHƯƠNG VI thì khi xí nghiệp liên doanh giải thể, hội đồng quản trị cần phải thực hiện những nghĩa vụ gì đối với người lao động và các bên liên quan? - mean=0.306
- Theo chỉ thị của UBNND TỈNH LÂM ĐỒNG, tình hình thực trạng quản lý hoạt động dạy nghề tại các tổ chức và doanh nghiệp trên địa bàn tỉnh hiện nay đang gặp phải những vấn đề gì? - mean=0.387
- CHƯƠNG VI nói về cái gì vậy ạ? - mean=0.406
