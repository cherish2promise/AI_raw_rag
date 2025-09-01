import os, re, json, hashlib, argparse, tempfile
from typing import List, Dict, Iterable, Tuple


GUIDE_DIR        = os.environ.get("GUIDE_DIR", "./guide")
PERSIST_DIR      = os.environ.get("PERSIST_DIR", "./chroma_store")
COLLECTION_NAME  = os.environ.get("COLLECTION_NAME", "guide_collection")

EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask") 
LLM_MODEL_NAME   = os.environ.get("LLM_MODEL_NAME", "beomi/gemma-ko-7b")

SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".csv"}

app = FastAPI()

# 공통 유틸
def normalize(text: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", (text or "")).strip()

def sent_split(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("—", "\n— ").replace("·", "\n· ").replace("-", "\n- ")
    parts = re.split(r"(?<=[\.!?])\s+|\n+", t)
    return [normalize(p) for p in parts if normalize(p)]

def make_chunks(texts: Iterable[str], chunk_size=900, chunk_overlap=150) -> List[str]:
    joined = normalize("\n".join([t for t in texts if t]))
    if not joined:
        return []
    out, n, i = [], len(joined), 0
    while i < n:
        j = min(i + chunk_size, n)
        out.append(joined[i:j].strip())
        if j == n: break
        i = max(0, j - chunk_overlap)
    return [c for c in out if c]


# 키포인트 스코어링

KEYWORDS = [
    "정의","용어","모델","개념","목적",
    "단계","절차","사전 검토","비식별","적정성 평가","사후 관리",
    "위험","재식별","보호","보안","평가","통제","정책","지침","가이드",
    "가명화","총계화","마스킹","일반화","무작위화","암호화",
    "k-익명성","l-다양성","t-근접성","차등 프라이버시",
    "공개","반공개","비공개","데이터 공개 모델",
    "요구사항","점검항목","관리적","기술적","물리적","ISMS-P","ISMSP"
]
BULLET_STARTS = ("—","-","·","•","*","▶","▪")

def score_sentence(s: str) -> int:
    score = 0
    for kw in KEYWORDS:
        if kw in s: score += 2
    if s.strip().startswith(BULLET_STARTS): score += 2
    if re.search(r"(정의|권장|해야|금지|평가|모델|단계|위험|보호|요구사항|가이드|지침)", s): score += 1
    wc = len(s)
    if wc < 8: score -= 1
    if wc > 240: score -= 1
    return score

def extract_keypoints(texts: Iterable[str], topn=7) -> List[Tuple[str,int]]:
    sents = []
    for t in texts:
        sents.extend(sent_split(t))
    seen, uniq = set(), []
    for s in sents:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    scored = [(s, score_sentence(s)) for s in uniq]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(s, sc) for s, sc in scored[:topn] if sc > 0]


# 파서: DOCX
from docx import Document
HEADING_KEYWORDS = ["개요","적용범위","인용","용어","정의","약어","비식별","유용성","부속서","해설","가이드","지침"]
HEADING_PATTERN = re.compile(r"^\s*(부속서\s*[A-Z]\b|[0-9]+(\.[0-9]+)*\b|[IVX]+\.)")

def _is_heading(paragraph) -> bool:
    name = (paragraph.style.name or "").lower()
    txt = normalize(paragraph.text)
    if not txt: return False
    if "heading" in name: return True
    if any(k in txt for k in HEADING_KEYWORDS): return True
    if HEADING_PATTERN.search(txt): return True
    if len(txt) <= 25 and not txt.startswith(BULLET_STARTS): return True
    return False

def parse_docx(path: str) -> List[Dict]:
    doc = Document(path)
    sections, current = [], {"title":"문서 시작","content":[]}
    for p in doc.paragraphs:
        txt = normalize(p.text)
        if not txt: continue
        if _is_heading(p):
            if current["content"]:
                sections.append(current)
            current = {"title": txt, "content": []}
        else:
            current["content"].append(txt)
    if current["content"]: sections.append(current)

    records = []
    for si, sec in enumerate(sections):
        for ci, chunk in enumerate(make_chunks(sec["content"])):
            records.append({
                "text": chunk,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "docx",
                    "section_title": sec["title"],
                    "section_index": si,
                    "chunk_index": ci,
                    "is_keypoint": False,
                    "score": None
                }
            })
        for kj, (kp, sc) in enumerate(extract_keypoints(sec["content"], topn=7)):
            records.append({
                "text": kp,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "docx",
                    "section_title": sec["title"],
                    "section_index": si,
                    "chunk_index": f"kp-{kj}",
                    "is_keypoint": True,
                    "score": int(sc)
                }
            })
    return records

# -----------------------------
# 파서: PDF
# -----------------------------
import pdfplumber

def parse_pdf(path: str) -> List[Dict]:
    records = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            txt = normalize(page.extract_text() or "")
            if not txt: continue
            paragraphs = [normalize(t) for t in re.split(r"\n{2,}", txt) if normalize(t)]
            for ci, chunk in enumerate(make_chunks(paragraphs)):
                records.append({
                    "text": chunk,
                    "meta": {
                        "source_path": os.path.abspath(path),
                        "file_type": "pdf",
                        "page": pno,
                        "chunk_index": ci,
                        "is_keypoint": False,
                        "score": None
                    }
                })
            for kj, (kp, sc) in enumerate(extract_keypoints(paragraphs, topn=7)):
                records.append({
                    "text": kp,
                    "meta": {
                        "source_path": os.path.abspath(path),
                        "file_type": "pdf",
                        "page": pno,
                        "chunk_index": f"kp-{kj}",
                        "is_keypoint": True,
                        "score": int(sc)
                    }
                })
    return records

# -----------------------------
# 파서: Excel/CSV
# -----------------------------
import pandas as pd

def row_to_text(headers: List[str], row: List) -> str:
    parts = []
    for h, v in zip(headers, row):
        hv = "" if pd.isna(v) else str(v)
        if hv != "":
            parts.append(f"{h}: {hv}")
    return " ; ".join(parts)

def parse_xlsx(path: str) -> List[Dict]:
    records = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet).fillna("")
        headers = [str(c) for c in df.columns]
        sheet_texts = []
        for ridx, row in enumerate(df.itertuples(index=False), start=0):
            txt = row_to_text(headers, list(row))
            if not txt: continue
            sheet_texts.append(txt)
            for ci, chunk in enumerate(make_chunks([txt], chunk_size=700, chunk_overlap=100)):
                records.append({
                    "text": chunk,
                    "meta": {
                        "source_path": os.path.abspath(path),
                        "file_type": "xlsx",
                        "sheet": sheet,
                        "row": ridx,
                        "chunk_index": ci,
                        "is_keypoint": False,
                        "score": None
                    }
                })
        for kj, (kp, sc) in enumerate(extract_keypoints(sheet_texts, topn=10)):
            records.append({
                "text": kp,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "xlsx",
                    "sheet": sheet,
                    "chunk_index": f"kp-{kj}",
                    "is_keypoint": True,
                    "score": int(sc)
                }
            })
    return records

def parse_csv_doc(path: str) -> List[Dict]:
    """문서성 CSV(가이드/규정 데이터)를 인덱싱용으로 파싱"""
    records = []
    df = pd.read_csv(path).fillna("")
    headers = [str(c) for c in df.columns]
    all_texts = []
    for ridx, row in enumerate(df.itertuples(index=False), start=0):
        txt = row_to_text(headers, list(row))
        if not txt: continue
        all_texts.append(txt)
        for ci, chunk in enumerate(make_chunks([txt], chunk_size=700, chunk_overlap=100)):
            records.append({
                "text": chunk,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "csv",
                    "row": ridx,
                    "chunk_index": ci,
                    "is_keypoint": False,
                    "score": None
                }
            })
    for kj, (kp, sc) in enumerate(extract_keypoints(all_texts, topn=10)):
        records.append({
            "text": kp,
            "meta": {
                "source_path": os.path.abspath(path),
                "file_type": "csv",
                "chunk_index": f"kp-{kj}",
                "is_keypoint": True,
                "score": int(sc)
            }
        })
    return records

# -----------------------------
# 파일 스캔/해시/파싱 스위치
# -----------------------------
def file_fingerprint(path: str) -> str:
    st = os.stat(path)
    base = f"{os.path.abspath(path)}::{st.st_size}::{int(st.st_mtime)}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def list_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_EXT:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def parse_by_type(path: str) -> List[Dict]:
    lp = path.lower()
    if lp.endswith(".pdf"):  return parse_pdf(path)
    if lp.endswith(".docx"): return parse_docx(path)
    if lp.endswith(".xlsx") or lp.endswith(".xls"): return parse_xlsx(path)
    if lp.endswith(".csv"):  return parse_csv_doc(path)
    if lp.endswith(".txt") or lp.endswith(".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        texts = [normalize(t) for t in re.split(r"\n{2,}", txt) if normalize(t)]
        recs = []
        for ci, chunk in enumerate(make_chunks(texts)):
            recs.append({
                "text": chunk,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "text",
                    "chunk_index": ci,
                    "is_keypoint": False,
                    "score": None
                }
            })
        for kj, (kp, sc) in enumerate(extract_keypoints(texts, topn=7)):
            recs.append({
                "text": kp,
                "meta": {
                    "source_path": os.path.abspath(path),
                    "file_type": "text",
                    "chunk_index": f"kp-{kj}",
                    "is_keypoint": True,
                    "score": int(sc)
                }
            })
        return recs
    return []

def stable_id(text: str, meta: Dict) -> str:
    base = json.dumps({"t": text, "m": meta}, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(base.encode("utf-8")).hexdigest()

# -----------------------------
# Chroma: 임베딩 & 컬렉션 (전역 캐싱)
# -----------------------------
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def embedding_dim_of(model_name: str) -> int:
    m = SentenceTransformer(model_name)
    try:
        return m.get_sentence_embedding_dimension()
    except Exception:
        import numpy as np
        return int(np.array(m.encode(["dim-check"], convert_to_numpy=True)).shape[-1])

def get_chroma_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
    target_dim = int(embedding_dim_of(EMBEDDING_MODEL))
    try:
        col = client.get_collection(COLLECTION_NAME)
        meta = col.metadata or {}
        existing_dim = meta.get("embedding_dim")
        if existing_dim is None or int(existing_dim) != target_dim:
            client.delete_collection(COLLECTION_NAME)
            col = client.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space":"cosine","embedding_dim":target_dim,"embedding_model":EMBEDDING_MODEL}
            )
    except Exception:
        col = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space":"cosine","embedding_dim":target_dim,"embedding_model":EMBEDDING_MODEL}
        )
    return col

def upsert_records(col, model, records: List[Dict], batch_size=64):
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        texts = [r["text"] for r in batch]
        metas = [r["meta"] for r in batch]
        ids   = [stable_id(t, m) for t, m in zip(texts, metas)]
        embs  = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        col.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=embs.tolist())

# -----------------------------
# 인덱싱(변경분만)
# -----------------------------
def build_or_update_index():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    col = get_chroma_collection()
    model = get_embed_model()

    manifest_path = os.path.join(PERSIST_DIR, f"{COLLECTION_NAME}_manifest.json")
    try:
        manifest = json.load(open(manifest_path, "r", encoding="utf-8"))
    except Exception:
        manifest = {}

    files = list_files(GUIDE_DIR)
    to_process = []
    for p in files:
        fp = file_fingerprint(p)
        if manifest.get(p) != fp:
            to_process.append((p, fp))

    if not to_process:
        print("✅ 인덱스 최신 상태 (변경 없음)")
        return

    all_records = []
    for p, fp in to_process:
        print(f"[*] 파싱: {p}")
        recs = parse_by_type(p)
        print(f"    → 레코드 {len(recs)}")
        all_records.extend(recs)

    if all_records:
        print(f"[*] 총 {len(all_records)}개 레코드 업서트 중…")
        upsert_records(col, model, all_records)
        for p, fp in to_process:
            manifest[p] = fp
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("✅ 인덱스 갱신 완료")
    else:
        print("변경 파일은 있었으나 생성 레코드가 없습니다(파서 출력 0).")

# -----------------------------
# 검색 & 응답 구성
# -----------------------------
def answer(query: str, topk_key=4, topk_chunk=4) -> Dict:
    col = get_chroma_collection()
    # 키포인트 우선(여유있게 더 뽑아서 병합 시 상위만 사용)
    res_kp = col.query(query_texts=[query], n_results=topk_key*2, where={"is_keypoint": True})
    res_ck = col.query(query_texts=[query], n_results=topk_chunk)

    # 병합(키포인트에 가중)
    seen, picks = set(), []
    def push(res, boost=1.0):
        if not res or not res.get("ids"): return
        ids = res["ids"][0]; docs = res["documents"][0]; metas = res["metadatas"][0]
        dists = res.get("distances", [[None]*len(ids)])[0]
        for i, rid in enumerate(ids):
            if rid in seen: continue
            seen.add(rid)
            picks.append({
                "id": rid, "text": docs[i], "meta": metas[i],
                "distance": dists[i], "boost": boost
            })
    push(res_kp, 1.5)
    push(res_ck, 1.0)

    def sort_key(x):
        d = x["distance"] if x["distance"] is not None else 1.0
        return d / x["boost"]
    picks.sort(key=sort_key)

    max_take = max(topk_key, topk_chunk)
    snippets = []
    for h in picks[: max_take]:
        m = h["meta"]; src = os.path.basename(m.get("source_path",""))
        where = ""
        if m.get("file_type")=="pdf" and m.get("page"): where = f"(p.{m['page']})"
        if m.get("file_type")=="docx" and m.get("section_title"): where = f"({m['section_title']})"
        if m.get("file_type")=="xlsx" and m.get("sheet") is not None: where = f"({m['sheet']})"
        if m.get("file_type")=="csv" and m.get("row") is not None: where = f"(row {m['row']})"
        snippets.append({
            "source": src, "where": where, "is_keypoint": bool(m.get("is_keypoint",False)), "text": h["text"]
        })

    bullets = [s["text"] for s in snippets if s["is_keypoint"]][:5]
    summary = " / ".join(bullets) if bullets else "관련 스니펫을 찾았습니다. 아래를 참고하세요."

    return {"query": query, "summary": summary, "snippets": snippets}

# -----------------------------
# RAG 생성 결합 (옵션)
# -----------------------------
RAG_PROMPT = """당신은 신중한 전문 조교입니다. 아래 '검색 스니펫'만 근거로 삼아 간결하고 정확한 한국어 답변을 작성하세요.
- 확실한 내용만 말하고, 불확실하면 "자료상 확인 불가"라고 명시
- 중요한 항목은 번호 목록으로 정리
- 답변 끝에 참고 출처 파일명과 위치를 나열

[질문]
{question}

[검색 스니펫]
{context}
"""

def format_context(snippets: List[Dict], max_chars=3000) -> str:
    lines, used = [], 0
    for s in snippets:
        prefix = f"[{s['source']}{(' ' + s['where']) if s['where'] else ''}] "
        seg = prefix + s["text"].replace("\n", " ")
        if used + len(seg) > max_chars: break
        lines.append(seg); used += len(seg)
    return "\n".join(lines)

# Lazy LLM 로더 (4bit). GPU 환경 권장.
_llm_pipe = None

def get_llm():
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    _llm_pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return _llm_pipe

def _strip_prompt_from_output(prompt: str, generated_text: str) -> str:
    # 프롬프트가 포함되어 돌아오는 경우 제거
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()

def generate_with_llm(question: str, snippets: List[Dict]) -> str:
    try:
        ctx = format_context(snippets, max_chars=2500)
        prompt = RAG_PROMPT.format(question=question, context=ctx)
        pipe = get_llm()
        out = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id,
        )[0]["generated_text"]
        return _strip_prompt_from_output(prompt, out)
    except Exception:
        # 로딩 실패나 OOM이면 검색 결과만 반환하도록 빈 문자열
        return ""

# -----------------------------
# CSV 배치 처리(질문 CSV → 답변 CSV)
# -----------------------------
QUESTION_COL_CANDIDATES = ["q", "question", "질문", "Query", "Question"]

from datetime import datetime

def _pick_question_column(df: pd.DataFrame) -> str:
    cols = [str(c).strip() for c in df.columns]
    # 우선순위 매칭
    for cand in QUESTION_COL_CANDIDATES:
        for c in cols:
            if c.lower().replace(" ", "") == cand.lower().replace(" ", ""):
                return c
    return cols[0]

def _format_sources(snippets: List[Dict]) -> str:
    outs = []
    for s in snippets[:6]:
        where = s.get("where") or ""
        src = s.get("source") or ""
        outs.append(f"{src}{where}")
    return " ; ".join(outs)

def process_csv(input_csv: str, output_csv: str, generate: bool = True, output_mode: str = "submission"):
    """
    질문 CSV → 답변 CSV 배치 처리
    output_mode:
      - "submission": 입력에 ID/Question이 있을 때 출력은 반드시 [ID, Answer]
      - "rich": 디버깅/검토용 상세 컬럼 [row_index, query, retrieval_summary, final_answer, sources, error]
    """
    build_or_update_index()
    df = pd.read_csv(input_csv)

    # 질문 컬럼 및 ID 컬럼 탐색
    qcol = _pick_question_column(df)
    idcol = None
    for c in df.columns:
        if str(c).strip().lower() == "id":
            idcol = c
            break

    out_rows = []
    for idx, row in df.iterrows():
        q = str(row.get(qcol, "")).strip()
        _id = row.get(idcol, idx) if idcol is not None else idx
        if not q:
            if output_mode == "submission":
                out_rows.append({"ID": _id, "Answer": ""})
            else:
                out_rows.append({
                    "row_index": idx,
                    "query": "",
                    "retrieval_summary": "",
                    "final_answer": "",
                    "sources": "",
                    "error": "빈 질문"
                })
            continue
        try:
            ret = answer(q)
            final_text = generate_with_llm(q, ret["snippets"]) if generate else ""
            answer_text = final_text or ret.get("summary", "")

            if output_mode == "submission":
                out_rows.append({"ID": _id, "Answer": answer_text})
            else:
                out_rows.append({
                    "row_index": idx,
                    "query": ret["query"],
                    "retrieval_summary": ret["summary"],
                    "final_answer": final_text or "",
                    "sources": _format_sources(ret["snippets"]),
                    "error": ""
                })
        except Exception as e:
            if output_mode == "submission":
                out_rows.append({"ID": _id, "Answer": ""})
            else:
                out_rows.append({
                    "row_index": idx,
                    "query": q,
                    "retrieval_summary": "",
                    "final_answer": "",
                    "sources": "",
                    "error": f"{type(e).__name__}: {e}"
                })

    if output_mode == "submission":
        out_df = pd.DataFrame(out_rows, columns=["ID", "Answer"])  # 컬럼 순서 고정
    else:
        out_df = pd.DataFrame(out_rows, columns=[
            "row_index", "query", "retrieval_summary", "final_answer", "sources", "error"
        ])

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 생성 완료 → {output_csv}")

# -----------------------------
# FastAPI (전역 app) + 엔드포인트
# -----------------------------
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

class AskBody(BaseModel):
    q: str
    generate: bool = False

app = FastAPI(title="Guide RAG")

@app.on_event("startup")
def _startup_build_index():
    build_or_update_index()

@app.post("/ask")
def ask(body: AskBody):
    ret = answer(body.q)
    final_text = generate_with_llm(body.q, ret["snippets"]) if body.generate else ""
    return {
        "query": ret["query"],
        "retrieval_summary": ret["summary"],
        "snippets": ret["snippets"],
        "final_answer": final_text or None
    }

@app.post("/reindex")
def reindex():
    build_or_update_index()
    return {"status": "ok"}

@app.post("/ask-csv")
def ask_csv_api(file: UploadFile = File(...), generate: bool = True, output_mode: str = "submission"):
    """질문 CSV 업로드 → 답변 CSV 생성 (기본: 제출 포맷[ID, Answer])"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_in:
        tmp_in.write(file.file.read())
        tmp_in_path = tmp_in.name

    out_path = os.path.join(
        os.path.dirname(tmp_in_path),
        f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    process_csv(tmp_in_path, out_path, generate=generate, output_mode=output_mode)

    return {
        "status": "ok",
        "output_csv": out_path,
        "output_mode": output_mode
    }

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="guide 폴더 인덱싱만 수행하고 종료")
    ap.add_argument("--ask", type=str, default=None, help="질의문 텍스트")
    ap.add_argument("--generate", action="store_true", help="LLM 생성까지 수행")
    ap.add_argument("--ask-csv", type=str, default=None, help="질문 CSV 경로")
    ap.add_argument("--out", type=str, default=None, help="출력 CSV 경로 (미지정 시 자동 생성)")
    ap.add_argument("--output-mode", choices=["submission", "rich"], default="submission", help="CSV 출력 형식: submission=[ID, Answer], rich=디버깅용 상세")
    args = ap.parse_args()

    print(f"스캔 대상: {os.path.abspath(GUIDE_DIR)}")
    build_or_update_index()

    if args.ask_csv:
        out_path = args.out or f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        process_csv(args.ask_csv, out_path, generate=args.generate, output_mode=args.output_mode)
        return

    if args.ask:
        ret = answer(args.ask)
        print("=== 요약 ===")
        print(ret["summary"])
        print("=== 스니펫 ===")
        for i, s in enumerate(ret["snippets"], 1):
            flag = "★" if s["is_keypoint"] else "·"
            print(f"{i}. {flag} {s['source']} {s['where']}{s['text']}")
        if args.generate:
            gen = generate_with_llm(args.ask, ret["snippets"])
            print("=== 최종 답변(LLM) ===")
            print(gen)

if __name__ == "__main__":
    main()


# 설치 예시
# pip install "chromadb==1.0.17" "sentence-transformers>=3" torch pdfplumber python-docx pandas openpyxl fastapi uvicorn transformers accelerate bitsandbytes
# 실행 예시
# uvicorn guide_rag:app --host 0.0.0.0 --port 8088 --reload
# CLI 예시
# python guide_rag.py --build
# python guide_rag.py --ask "비식별 절차 요약" --generate
# python guide_rag.py --ask-csv questions.csv --generate --out answers.csv

# python guide_rag.py --ask-csv questions.csv --generate --out answers.csv
# pip install "chromadb==1.0.17" "sentence-transformers>=3" "torch"  pdfplumber python-docx pandas openpyxl fastapi uvicorn  transformers accelerate bitsandbytes
# uvicorn guide_rag:app --host 0.0.0.0 --port 8088 --reload