# Vietnamese Legal RAG

Hỏi–đáp pháp luật lao động Việt Nam bằng RAG (Retrieval-Augmented Generation):
tìm đúng điều luật liên quan rồi để Gemini trả lời kèm trích nguồn.

## Chạy thử ngay (không cần chuẩn bị dữ liệu)

Repo đã kèm sẵn corpus mẫu 300 văn bản (`data/processed/labor_corpus_sample.jsonl`), clone về là chạy được:

```bash
pip install -r requirements.txt
python app.py "Người lao động đơn phương chấm dứt hợp đồng cần báo trước bao lâu?"
```

Không có Gemini key thì CLI chỉ in các nguồn luật tìm được. Để sinh câu trả lời:

```bash
cp .env.example .env   # điền GEMINI_API_KEY hoặc GEMINI_API_KEYS (nhiều key, cách nhau dấu phẩy)
```

Demo web:

```bash
streamlit run app/streamlit_app.py
```

## Cách hoạt động (1 phút)

1. **Dữ liệu**: văn bản pháp luật lao động lấy từ [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents), cắt thành chunk theo từng "Điều".
2. **Retrieval** (pipeline được chọn `hybrid_rrf_cross_encoder_mmr`):
   - BM25 (từ khóa) + FAISS vector search (`intfloat/multilingual-e5-small`) chạy song song
   - Gộp 2 danh sách bằng RRF (Reciprocal Rank Fusion)
   - Cross-Encoder (`mmarco-mMiniLMv2`) chấm lại độ liên quan từng cặp (câu hỏi, chunk)
   - MMR loại các chunk trùng lặp để context đa dạng
3. **Generation**: Gemini (`gemini-3.1-flash-lite`) trả lời chỉ dựa trên context, trích nguồn dạng [S1], [S2]. Nhiều API key được xoay vòng kèm rate-limit và retry (`src/generation/gemini_client.py`).

CLI/Streamlit chạy đúng pipeline này (ưu tiên `labor_corpus.jsonl` đầy đủ nếu có, không thì dùng corpus mẫu; embedding được cache trong `.cache/`).

## Đánh giá (vì sao chọn pipeline này)

Quy trình 2 bước, chạy lại được bằng script:

```bash
python scripts/make_sample_corpus.py    # tạo corpus mẫu 300 văn bản từ HuggingFace
python scripts/build_benchmark.py       # Gemini sinh 30 câu hỏi paraphrase + gold chunk labels
python scripts/run_retrieval_eval.py    # Recall@5 / MRR / nDCG@5 chính xác theo gold labels
python scripts/run_generation_eval.py   # RAGAS: faithfulness / relevancy / correctness
```

Điểm quan trọng của benchmark:

- Mỗi câu hỏi được **paraphrase** khỏi câu chữ luật gốc (hỏi như người lao động thật) nên BM25 thuần không dễ ăn điểm — các method tách bạch rõ thay vì cùng đạt 1.0.
- Mỗi câu hỏi lưu **gold_chunk_id** (chunk gốc sinh ra nó) nên Recall/MRR là số đo chính xác, không phải ước lượng trùng từ.
- RAGAS dùng Gemini làm judge, embedding chạy local; dòng nào lỗi (NaN) được **retry với key khác** nên coverage gần đủ 30/30 câu.

Kết quả mới nhất: `reports/retrieval_evaluation.md` và `reports/ragas_evaluation.md`.
Các notebook trong `notebooks/` ghi lại quá trình thí nghiệm ban đầu trên corpus đầy đủ.

## Cấu trúc thư mục

```text
app.py                  # CLI hỏi đáp
app/streamlit_app.py    # demo web
src/data/               # schema, tải dataset, làm sạch, chunking theo Điều
src/retrieval/          # RetrievalPipeline: bm25 / vector / hybrid / rrf / cross-encoder / mmr
src/generation/         # Gemini client (xoay key + retry), prompt, RAG pipeline
scripts/                # tạo corpus mẫu, benchmark, chạy 2 bài đánh giá
notebooks/              # thí nghiệm gốc (01 data, 02 retrieval, 03 ragas)
reports/                # benchmark + kết quả đánh giá
tests/                  # unit test (chạy trong CI)
```

## Test & CI

```bash
python -m unittest discover -s tests -v
```

GitHub Actions (`.github/workflows/ci.yml`) chạy test trên mỗi push/PR.
Cần chạy notebook hoặc script đánh giá thì cài thêm: `pip install -r requirements-eval.txt`.

## Lưu ý pháp lý

Demo chỉ phục vụ tra cứu thông tin, không thay thế tư vấn pháp lý chuyên nghiệp.
