# RAGAS Answer Evaluation

- Backend: `ragas_evaluate`
- Cached artifact used: `True`
- Cache reason: `cached_answers`
- Answers: `30`
- Retrieval method: `hybrid`
- Model: `gemini-3.1-flash-lite`

| Metric | Score |
|---|---:|
| context_precision | 0.467 |
| context_recall | 0.517 |
| faithfulness | 0.798 |
| answer_relevancy | 0.432 |
| answer_correctness | unavailable |

## Figures

- `reports/figures/ragas_answer_metrics.png`
- `reports/figures/ragas_per_question_heatmap.png`

## Weakest Cases

- Văn bản này hướng dẫn thực hiện các quy định về lao động dựa trên nghị định nào của Chính phủ? - mean=0.000
- Cty em phai chi bao nhieu % quy luong cho LD-TB & XH de dao tao nghe? - mean=0.000
- Cho em hoi la cai Sở Lao động liên khu no co nhiem vu gi trong viec giai quyet mau thuan giua chu voi cong nhan khong a? - mean=0.000
- Sở Lao động liên khu có nhiệm vụ gì trong việc giải quyết mâu thuẫn giữa chủ và thợ? - mean=0.000
- Cho em hoi la UBNN tinh co trach nhiem gi trong viec phoi hop voi nganh LĐ-TB & XH de ho tro viec dao tao nghe va viec lam cho nguoi tan tat khong a? - mean=0.000
