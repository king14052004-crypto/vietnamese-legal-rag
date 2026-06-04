import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.data.chunking import chunk_documents
from src.data.load_dataset import load_documents_from_jsonl
from src.evaluation.config import (
    CORPUS_PATH,
    RAGAS_TESTSET_JSON,
    RAGAS_TESTSET_MD,
    REPORTS_DIR,
    TESTSET_CANDIDATES,
    TESTSET_MIN_ACCEPTED,
)
from src.evaluation.gemini_env import first_gemini_key as _first_gemini_key
from src.evaluation.gemini_env import has_gemini_key as _has_gemini_key
from src.evaluation.ragas_runtime import build_ragas_embeddings, build_ragas_llm


@dataclass
class TestsetItem:
    id: str
    question: str
    reference: str
    expected_terms: list[str]
    topic: str
    source: str


FALLBACK_TESTSET = [
    TestsetItem("labor_001", "Người lao động đơn phương chấm dứt hợp đồng lao động cần báo trước bao lâu?", "Thời hạn báo trước phụ thuộc loại hợp đồng và trường hợp chấm dứt. Người lao động thường phải báo trước theo thời hạn luật định trước khi đơn phương chấm dứt hợp đồng.", ["đơn phương chấm dứt", "hợp đồng lao động", "báo trước"], "termination", "fallback_curated_seed"),
    TestsetItem("labor_002", "Người sử dụng lao động đơn phương chấm dứt hợp đồng cần báo trước trong trường hợp nào?", "Người sử dụng lao động chỉ được đơn phương chấm dứt trong các trường hợp luật cho phép và phải báo trước theo thời hạn tương ứng, trừ trường hợp được miễn báo trước.", ["người sử dụng lao động", "đơn phương chấm dứt", "báo trước"], "termination", "fallback_curated_seed"),
    TestsetItem("labor_003", "Trường hợp nào người lao động được nhận trợ cấp thôi việc?", "Trợ cấp thôi việc áp dụng khi quan hệ lao động chấm dứt hợp lệ và người lao động đáp ứng điều kiện về thời gian làm việc, trừ các trường hợp không được hưởng theo luật.", ["trợ cấp thôi việc", "chấm dứt hợp đồng", "người lao động"], "benefits", "fallback_curated_seed"),
    TestsetItem("labor_004", "Trợ cấp mất việc khác trợ cấp thôi việc như thế nào?", "Trợ cấp mất việc thường gắn với lý do tổ chức, cơ cấu, công nghệ hoặc thu hẹp hoạt động; trợ cấp thôi việc gắn với các trường hợp chấm dứt hợp đồng thông thường.", ["trợ cấp mất việc", "trợ cấp thôi việc", "chấm dứt hợp đồng"], "benefits", "fallback_curated_seed"),
    TestsetItem("labor_005", "Doanh nghiệp có trách nhiệm gì về an toàn vệ sinh lao động?", "Doanh nghiệp phải bảo đảm điều kiện an toàn vệ sinh lao động, trang bị bảo hộ, phòng ngừa tai nạn lao động và thực hiện các nghĩa vụ theo quy định.", ["an toàn vệ sinh lao động", "bảo hộ lao động", "tai nạn lao động"], "safety", "fallback_curated_seed"),
    TestsetItem("labor_006", "Người lao động bị tai nạn lao động được hưởng quyền lợi gì?", "Người lao động bị tai nạn lao động có thể được khám chữa bệnh, giám định, trợ cấp hoặc bồi thường tùy mức độ và căn cứ pháp luật áp dụng.", ["tai nạn lao động", "trợ cấp", "giám định"], "safety", "fallback_curated_seed"),
    TestsetItem("labor_007", "Quy định về tiền lương tối thiểu áp dụng cho người lao động như thế nào?", "Tiền lương tối thiểu là mức sàn trả lương theo quy định, thường phân theo vùng hoặc đối tượng và là căn cứ bảo vệ thu nhập tối thiểu của người lao động.", ["lương tối thiểu", "tiền lương", "người lao động"], "salary", "fallback_curated_seed"),
    TestsetItem("labor_008", "Người sử dụng lao động có được trả lương thấp hơn lương tối thiểu không?", "Người sử dụng lao động không được trả lương thấp hơn mức lương tối thiểu áp dụng cho công việc và địa bàn theo quy định.", ["lương tối thiểu", "trả lương", "người sử dụng lao động"], "salary", "fallback_curated_seed"),
    TestsetItem("labor_009", "Làm thêm giờ được quản lý theo nguyên tắc nào?", "Làm thêm giờ phải tuân thủ giới hạn, điều kiện và chế độ trả lương làm thêm theo quy định pháp luật lao động.", ["làm thêm giờ", "tiền lương", "thời giờ làm việc"], "working_time", "fallback_curated_seed"),
    TestsetItem("labor_010", "Thời giờ làm việc và thời giờ nghỉ ngơi được quy định nhằm mục đích gì?", "Quy định về thời giờ làm việc và nghỉ ngơi nhằm giới hạn thời gian lao động, bảo vệ sức khỏe và quyền nghỉ ngơi của người lao động.", ["thời giờ làm việc", "thời giờ nghỉ ngơi", "người lao động"], "working_time", "fallback_curated_seed"),
    TestsetItem("labor_011", "Người lao động nước ngoài cần điều kiện gì để làm việc tại Việt Nam?", "Người lao động nước ngoài thường cần đáp ứng điều kiện chuyên môn, hồ sơ pháp lý và giấy phép lao động hoặc trường hợp miễn giấy phép theo quy định.", ["lao động nước ngoài", "giấy phép lao động", "Việt Nam"], "foreign_worker", "fallback_curated_seed"),
    TestsetItem("labor_012", "Khi nào người lao động nước ngoài phải có giấy phép lao động?", "Người lao động nước ngoài làm việc tại Việt Nam phải có giấy phép lao động nếu không thuộc trường hợp được miễn theo quy định.", ["người lao động nước ngoài", "giấy phép lao động", "miễn"], "foreign_worker", "fallback_curated_seed"),
    TestsetItem("labor_013", "Kỷ luật sa thải người lao động được áp dụng trong trường hợp nào?", "Sa thải là hình thức kỷ luật lao động nghiêm khắc, chỉ áp dụng khi có căn cứ vi phạm thuộc trường hợp luật hoặc nội quy hợp pháp quy định.", ["kỷ luật lao động", "sa thải", "người lao động"], "discipline", "fallback_curated_seed"),
    TestsetItem("labor_014", "Khi xử lý kỷ luật lao động cần bảo đảm nguyên tắc gì?", "Xử lý kỷ luật lao động phải có căn cứ, đúng trình tự, bảo đảm quyền tham gia hoặc trình bày của người lao động theo quy định.", ["kỷ luật lao động", "trình tự", "người lao động"], "discipline", "fallback_curated_seed"),
    TestsetItem("labor_015", "Bảo hiểm thất nghiệp hỗ trợ người lao động như thế nào?", "Bảo hiểm thất nghiệp hỗ trợ người lao động khi mất việc thông qua trợ cấp, tư vấn, giới thiệu việc làm hoặc hỗ trợ học nghề theo quy định.", ["bảo hiểm thất nghiệp", "trợ cấp", "việc làm"], "insurance", "fallback_curated_seed"),
    TestsetItem("labor_016", "Bảo hiểm xã hội có vai trò gì trong quan hệ lao động?", "Bảo hiểm xã hội bảo vệ người lao động trước các rủi ro như ốm đau, thai sản, tai nạn lao động, mất sức lao động, hưu trí hoặc tử tuất.", ["bảo hiểm xã hội", "người lao động", "thai sản"], "insurance", "fallback_curated_seed"),
    TestsetItem("labor_017", "Lao động nữ được bảo vệ như thế nào trong pháp luật lao động?", "Pháp luật lao động có các quy định bảo vệ lao động nữ, đặc biệt về thai sản, điều kiện làm việc, thời giờ làm việc và chống phân biệt đối xử.", ["lao động nữ", "thai sản", "điều kiện làm việc"], "female_worker", "fallback_curated_seed"),
    TestsetItem("labor_018", "Người lao động nghỉ thai sản được hưởng quyền lợi gì?", "Người lao động nghỉ thai sản được hưởng chế độ nghỉ và quyền lợi bảo hiểm hoặc tiền trợ cấp nếu đáp ứng điều kiện theo quy định.", ["thai sản", "nghỉ thai sản", "bảo hiểm"], "female_worker", "fallback_curated_seed"),
    TestsetItem("labor_019", "Công đoàn có vai trò gì trong doanh nghiệp?", "Công đoàn đại diện và bảo vệ quyền, lợi ích hợp pháp của người lao động, tham gia thương lượng và phối hợp giải quyết vấn đề lao động.", ["công đoàn", "người lao động", "đại diện"], "union", "fallback_curated_seed"),
    TestsetItem("labor_020", "Thỏa ước lao động tập thể có ý nghĩa gì?", "Thỏa ước lao động tập thể ghi nhận các thỏa thuận giữa tập thể lao động và người sử dụng lao động về điều kiện lao động và quyền lợi liên quan.", ["thỏa ước lao động tập thể", "tập thể lao động", "người sử dụng lao động"], "collective", "fallback_curated_seed"),
    TestsetItem("labor_021", "Tranh chấp lao động tập thể được giải quyết như thế nào?", "Tranh chấp lao động tập thể thường được giải quyết qua thương lượng, hòa giải, trọng tài hoặc cơ quan có thẩm quyền theo trình tự luật định.", ["tranh chấp lao động", "tập thể", "hòa giải"], "dispute", "fallback_curated_seed"),
    TestsetItem("labor_022", "Tranh chấp lao động cá nhân khác tranh chấp lao động tập thể ở điểm nào?", "Tranh chấp cá nhân liên quan quyền lợi của một hoặc một số người lao động; tranh chấp tập thể liên quan quyền và lợi ích của tập thể lao động.", ["tranh chấp lao động", "cá nhân", "tập thể"], "dispute", "fallback_curated_seed"),
    TestsetItem("labor_023", "Hợp đồng lao động cần thể hiện những nội dung chính nào?", "Hợp đồng lao động cần thể hiện công việc, địa điểm, thời hạn, tiền lương, thời giờ làm việc, điều kiện lao động và quyền nghĩa vụ cơ bản.", ["hợp đồng lao động", "tiền lương", "thời giờ làm việc"], "contract", "fallback_curated_seed"),
    TestsetItem("labor_024", "Hợp đồng lao động xác định thời hạn khác hợp đồng không xác định thời hạn thế nào?", "Hợp đồng xác định thời hạn có thời điểm kết thúc; hợp đồng không xác định thời hạn không ấn định thời điểm chấm dứt.", ["hợp đồng xác định thời hạn", "không xác định thời hạn", "hợp đồng lao động"], "contract", "fallback_curated_seed"),
    TestsetItem("labor_025", "Thử việc có được chấm dứt mà không báo trước không?", "Trong thời gian thử việc, mỗi bên có thể chấm dứt thỏa thuận thử việc theo điều kiện luật định, thường không phải báo trước nếu thử việc không đạt.", ["thử việc", "chấm dứt", "báo trước"], "probation", "fallback_curated_seed"),
    TestsetItem("labor_026", "Dạy nghề và đào tạo nghề liên quan gì đến quan hệ lao động?", "Dạy nghề và đào tạo nghề giúp chuẩn bị hoặc nâng cao kỹ năng lao động; một số trường hợp có cam kết đào tạo và trách nhiệm hoàn trả chi phí.", ["dạy nghề", "đào tạo nghề", "người lao động"], "training", "fallback_curated_seed"),
    TestsetItem("labor_027", "Người lao động có quyền khiếu nại khi bị xâm phạm quyền lợi không?", "Người lao động có quyền khiếu nại hoặc yêu cầu cơ quan có thẩm quyền giải quyết khi quyền, lợi ích hợp pháp bị xâm phạm.", ["khiếu nại", "quyền lợi", "người lao động"], "rights", "fallback_curated_seed"),
    TestsetItem("labor_028", "Nội quy lao động có vai trò gì trong xử lý kỷ luật?", "Nội quy lao động là căn cứ quan trọng để quản lý lao động và xử lý kỷ luật nếu nội quy hợp pháp, được ban hành đúng quy định.", ["nội quy lao động", "kỷ luật", "xử lý"], "discipline", "fallback_curated_seed"),
    TestsetItem("labor_029", "Người lao động có được bảo hộ lao động khi làm việc nguy hiểm không?", "Người lao động làm việc trong điều kiện nguy hiểm hoặc độc hại phải được bảo hộ lao động và áp dụng biện pháp phòng ngừa rủi ro.", ["bảo hộ lao động", "độc hại", "nguy hiểm"], "safety", "fallback_curated_seed"),
    TestsetItem("labor_030", "Doanh nghiệp phải làm gì khi thay đổi cơ cấu dẫn đến mất việc?", "Khi thay đổi cơ cấu, công nghệ hoặc tổ chức dẫn đến mất việc, doanh nghiệp phải thực hiện nghĩa vụ thông báo, tham khảo, hỗ trợ hoặc trợ cấp theo quy định.", ["thay đổi cơ cấu", "mất việc", "trợ cấp"], "termination", "fallback_curated_seed"),
]


def has_gemini_key() -> bool:
    return _has_gemini_key()


def first_gemini_key() -> str | None:
    return _first_gemini_key()


def load_testset(path: Path = RAGAS_TESTSET_JSON) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) and "items" in data else data


def testset_generated_with_ragas(path: Path = RAGAS_TESTSET_JSON) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(isinstance(data, dict) and data.get("generated_with_ragas"))


def write_testset(items: list[dict], path: Path = RAGAS_TESTSET_JSON, generated_with_ragas: bool = False) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_with_ragas": generated_with_ragas,
        "candidate_target": TESTSET_CANDIDATES,
        "accepted_count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# RAGAS Testset",
        "",
        f"- Generated with RAGAS TestsetGenerator: `{generated_with_ragas}`",
        f"- Accepted questions: `{len(items)}`",
        "",
        "| ID | Topic | Source | Question |",
        "|---|---|---|---|",
    ]
    for item in items:
        lines.append(f"| {item['id']} | {item['topic']} | {item['source']} | {item['question']} |")
    RAGAS_TESTSET_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fallback_testset() -> list[dict]:
    return [asdict(item) for item in FALLBACK_TESTSET]


def _to_langchain_documents(max_docs: int = 12):
    from langchain_core.documents import Document

    documents = load_documents_from_jsonl(str(CORPUS_PATH))
    chunks = chunk_documents(documents)
    selected = []
    seen_ids = set()
    for item in fallback_testset():
        terms = item["expected_terms"]
        for chunk in chunks:
            text = f"{chunk.title} {chunk.text}".lower()
            if chunk.chunk_id in seen_ids or not any(term.lower() in text for term in terms):
                continue
            seen_ids.add(chunk.chunk_id)
            selected.append(
                Document(
                    page_content=chunk.text[:600],
                    metadata={
                        "doc_id": chunk.doc_id,
                        "title": chunk.title,
                        "chunk_id": chunk.chunk_id,
                        "topic": item["topic"],
                    },
                )
            )
            break
        if len(selected) >= max_docs:
            break
    return selected


def generate_ragas_testset(size: int = TESTSET_CANDIDATES) -> list[dict]:
    if not first_gemini_key():
        raise RuntimeError("Set GEMINI_API_KEY, GEMINI_API_KEYS, or GOOGLE_API_KEY to run RAGAS TestsetGenerator")

    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

    llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()
    llm_context = (
        "Generate Vietnamese labor-law benchmark questions and reference answers. "
        "Questions must be practical, concise, and answerable only from the provided legal context."
    )
    personas = [
        Persona(name="Worker", role_description="An employee asking about rights and obligations under Vietnamese labor law."),
        Persona(name="Employer", role_description="An employer or HR manager checking legal compliance for labor-law situations."),
        Persona(name="LegalAdvisor", role_description="A legal advisor reviewing Vietnamese labor-law questions for a client."),
    ]
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=llm, llm_context=llm_context), 1.0),
    ]
    generator = TestsetGenerator(llm=llm, embedding_model=embeddings, persona_list=personas, llm_context=llm_context)
    testset = generator.generate_with_langchain_docs(
        _to_langchain_documents(),
        testset_size=size,
        query_distribution=query_distribution,
        raise_exceptions=False,
    )
    frame = testset.to_pandas()
    items = []
    for idx, row in frame.iterrows():
        question = str(row.get("user_input") or row.get("question") or "").strip()
        reference = str(row.get("reference") or row.get("reference_contexts") or "").strip()
        if not question or len(question.split()) < 5:
            continue
        items.append(
            {
                "id": f"ragas_{idx + 1:03d}",
                "question": question,
                "reference": reference or question,
                "expected_terms": [],
                "topic": str(row.get("synthesizer_name") or "ragas_generated"),
                "source": "ragas_testset_generator",
            }
        )
    return items[:TESTSET_MIN_ACCEPTED]


def ensure_testset(prefer_ragas: bool = True) -> list[dict]:
    existing = load_testset()
    if len(existing) >= TESTSET_MIN_ACCEPTED and (testset_generated_with_ragas() or not prefer_ragas or not has_gemini_key()):
        return existing
    if prefer_ragas and has_gemini_key():
        try:
            generated = generate_ragas_testset()
            if len(generated) >= TESTSET_MIN_ACCEPTED:
                write_testset(generated, generated_with_ragas=True)
                return generated
        except Exception as exc:
            print(f"RAGAS TestsetGenerator failed, using fallback curated seed: {exc}")
    items = fallback_testset()
    write_testset(items, generated_with_ragas=False)
    return items


def main() -> None:
    items = ensure_testset()
    payload = json.loads(RAGAS_TESTSET_JSON.read_text(encoding="utf-8"))
    print(f"testset items={len(items)} generated_with_ragas={payload['generated_with_ragas']}")


if __name__ == "__main__":
    main()
