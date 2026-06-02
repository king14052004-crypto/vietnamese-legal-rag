# RAGAS-style Evaluation

Evaluator: AI Studio Gemini API via `google-genai`.

| Metric | Score |
|---|---:|
| Faithfulness | 0.950 |
| Answer relevancy | 0.950 |
| Context precision | 0.812 |

## Per-sample notes

- **Người lao động đơn phương chấm dứt hợp đồng lao động cần báo trước bao lâu?** — F=1.00, R=1.00, CP=1.00. Câu trả lời trích xuất chính xác các thông tin về thời hạn báo trước từ context [3] và [5]. Các thông tin được cung cấp phù hợp hoàn toàn với dữ liệu đầu vào.
- **Trường hợp nào người lao động được nhận trợ cấp thôi việc?** — F=1.00, R=1.00, CP=0.80. Câu trả lời bám sát nội dung được cung cấp trong các đoạn văn bản (context). Độ chính xác của context_precision bị trừ nhẹ do trong câu trả lời có trích dẫn các mục/điểm (S1, S3, S4) chưa được định danh rõ ràng trong các đoạn văn bản được cung cấp.
- **Doanh nghiệp có trách nhiệm gì về an toàn vệ sinh lao động?** — F=1.00, R=1.00, CP=1.00. Câu trả lời phản ánh chính xác và đầy đủ các trách nhiệm của doanh nghiệp dựa trên thông tin được cung cấp trong các đoạn context [1], [3], [4], [6]. Các trích dẫn [S1], [S3], [S4], [S6] được sử dụng hợp lý.
- **Quy định về tiền lương và lương tối thiểu của người lao động là gì?** — F=1.00, R=1.00, CP=1.00. Câu trả lời trích xuất chính xác thông tin từ các ngữ cảnh được cung cấp, có phân loại rõ ràng theo từng đối tượng và bổ sung cảnh báo cần thiết về tính thời điểm của dữ liệu.
- **Người lao động nước ngoài cần điều kiện gì để làm việc tại Việt Nam?** — F=1.00, R=1.00, CP=0.80. Câu trả lời trích xuất chính xác thông tin từ ngữ cảnh (S1, S4, S5). Tuy nhiên, ngữ cảnh chứa nhiều văn bản pháp luật cũ (năm 1996, 1999, 2000) gây nhiễu về mặt giá trị thời đại, dù mô hình đã xử lý khéo léo bằng cách thêm cảnh báo.
- **Kỷ luật sa thải người lao động được áp dụng trong trường hợp nào?** — F=0.80, R=0.70, CP=0.80. Câu trả lời chính xác dựa trên ngữ cảnh nhưng hơi lan man khi liệt kê các trường hợp 'đơn phương chấm dứt HĐLĐ' (vốn là vấn đề dân sự/quản lý) thay vì tập trung vào 'kỷ luật' (vốn là vấn đề xử phạt). Ngữ cảnh có nhắc đến cụm từ 'kỷ luật buộc thôi việc' nhưng văn bản không liệt kê chi tiết các hành vi cấu thành hình thức kỷ luật này, dẫn đến việc người trả lời phải suy diễn từ các mục khác.
- **Bảo hiểm thất nghiệp hỗ trợ người lao động như thế nào?** — F=1.00, R=1.00, CP=0.40. Câu trả lời trung thành với S1. Tuy nhiên, context chứa nhiều nhiễu (về bảo hiểm y tế và các quy định không liên quan) khiến độ chính xác của ngữ cảnh thấp.
- **Tranh chấp lao động tập thể được giải quyết như thế nào?** — F=0.80, R=0.90, CP=0.70. Câu trả lời bám sát nội dung được cung cấp từ các ngữ cảnh (S1, S4, S6). Tuy nhiên, phần trích dẫn [S6] trong câu trả lời không khớp chính xác với nội dung [S6] trong ngữ cảnh (chỉ nói về khiếu nại), và cấu trúc trình bày có phần chồng chéo giữa quy định chung và xí nghiệp vốn đầu tư nước ngoài.