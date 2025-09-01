"""
Law RAG (single-file)

기능
- rag_laws 폴더의 JSONL 파싱 → 조항/청크 생성
- guide 폴더의 PDF/DOCX/XLSX/XLS/TXT/MD/CSV/JSONL 파싱 → 청크 생성
- SentenceTransformer 임베딩 → Chroma Persistent 업서트
- 검색/요약 기반 질의 응답(기본은 요약, 선택적으로 LLM 사용 가능)
- CSV/XLSX 배치 처리 (--ask-csv) : 인코딩 자동 시도, 컬럼 자동 탐지, 20행마다 자동 저장
- FastAPI 조회 엔드포인트(/health, /collections, /peek, /search)

안전장치
- 경로 안정화(BASE_DIR 기준)
- HNSW 손상 자동 감지 → 컬렉션 재생성 후 친절한 에러 안내
- 빈 데이터/인코딩/저장 오류 친절 로그

실행 예
- 임베딩: python law_rag.py --embed
- 단일 질의: python law_rag.py --ask "질문"
- 배치: python law_rag.py --ask-csv .\questions.csv --output .\submission.csv
- 서버: uvicorn law_rag:app --reload

권장
- OneDrive/한글 경로 회피: $env:PERSIST_DIR="C:\chroma_store"
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Iterable, Tuple
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# =========================================
# FastAPI
# =========================================
app = FastAPI(title="Law RAG")

# =========================================
# 설정 (경로 안정화)
# =========================================
BASE_DIR = Path(__file__).resolve().parent

def _env_path(name: str, default_rel: str) -> str:
    p = os.environ.get(name)
    return str(Path(p).expanduser().resolve()) if p else str((BASE_DIR / default_rel).resolve())

LAW_DIR         = _env_path("LAW_DIR", "./rag_laws")       # JSONL 폴더
GUIDE_DIR       = _env_path("GUIDE_DIR", "./guide")        # 가이드 문서 폴더
PERSIST_DIR     = _env_path("PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "law_guide_collection")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")  # 768-d
LLM_MODEL_NAME  = os.environ.get("LLM_MODEL_NAME", "beomi/gemma-ko-7b")

SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".csv", ".jsonl"}

print(f"[PATH] LAW_DIR={LAW_DIR}")
print(f"[PATH] GUIDE_DIR={GUIDE_DIR}")
print(f"[PATH] PERSIST_DIR={PERSIST_DIR}")

# =========================================
# 공통 유틸
# =========================================
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

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

# =========================================
# 키포인트 스코어링 (선택적으로 사용 가능)
# =========================================
KEYWORDS = [
    "정의","용어","모델","개념","목적",
    "단계","절차","사전 검토","비식별","적정성 평가","사후 관리",
    "위험","재식별","보호","보안","평가","통제","정책","지침","가이드",
    "가명화","총계화","마스킹","일반화","무작위화","암호화",
    "k-익명성","l-다양성","t-근접성","차등 프라이버시",
    "공개","반공개","비공개","데이터 공개 모델",
    "요구사항","점검항목","관리적","기술적","물리적","ISMS-P","ISMSP",
    "법률","조항","규정","의무","권리","책임","처벌","벌칙","과태료",
    "개인정보","정보주체","처리자","동의","수집","이용","제공","파기",
    "전자금융","금융거래","보안","인증","암호화","접근통제"
]
BULLET_STARTS = ("—","-","·","•","*","▶","▪")

def score_sentence(s: str) -> int:
    score = 0
    for kw in KEYWORDS:
        if kw in s: score += 2
    if s.strip().startswith(BULLET_STARTS): score += 2
    if re.search(r"(정의|권장|해야|금지|평가|모델|단계|위험|보호|요구사항|가이드|지침|의무|권리|책임)", s): score += 1
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

# =========================================
# 법률 문서 파싱(JSONL)
# =========================================
def parse_law_jsonl(file_path: str) -> List[Dict]:
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                if doc.get('text'):
                    documents.append(doc)
            except json.JSONDecodeError:
                continue
    return documents

def process_law_documents() -> List[Dict]:
    if not os.path.isdir(LAW_DIR):
        print(f"[WARN] LAW_DIR not found: {LAW_DIR} → 법률 문서를 스킵합니다.")
        return []
    all_chunks: List[Dict] = []
    for filename in os.listdir(LAW_DIR):
        if not filename.endswith('.jsonl'):
            continue
        file_path = os.path.join(LAW_DIR, filename)
        print(f"Processing law file: {filename}")
        documents = parse_law_jsonl(file_path)
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            metadata = {
                'source': filename,
                'law_name': doc.get('law_name', ''),
                'article_no': doc.get('article_no', ''),
                'article_title': doc.get('article_title', ''),
                'chapter': doc.get('chapter', ''),
                'law_code': doc.get('law_code', ''),
                'tags': ','.join(doc.get('tags', [])) if isinstance(doc.get('tags', []), list) else str(doc.get('tags', "")),
                'keywords_hint': ','.join(doc.get('keywords_hint', [])) if isinstance(doc.get('keywords_hint', []), list) else str(doc.get('keywords_hint', "")),
                'doc_type': 'law',
            }
            chunks = make_chunks([text])
            for i, chunk in enumerate(chunks):
                chunk_md = dict(metadata)
                chunk_md['chunk_id'] = f"{doc.get('chunk_id', '')}_{i}"
                chunk_md['chunk_index'] = i
                all_chunks.append({'text': chunk, 'metadata': chunk_md})
    return all_chunks

# =========================================
# 가이드 문서 파싱
# =========================================
def parse_pdf(file_path: str) -> List[str]:
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return texts
    except Exception as e:
        print(f"PDF 파싱 오류 {file_path}: {e}")
        return []

def parse_docx(file_path: str) -> List[str]:
    try:
        from docx import Document
        doc = Document(file_path)
        return [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        print(f"DOCX 파싱 오류 {file_path}: {e}")
        return []

def parse_excel(file_path: str) -> List[str]:
    try:
        df = pd.read_excel(file_path)
        texts = []
        for col in df.columns:
            texts.extend([str(x) for x in df[col].dropna() if str(x).strip()])
        return texts
    except Exception as e:
        print(f"Excel 파싱 오류 {file_path}: {e}")
        return []

def parse_txt(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
    except Exception as e:
        print(f"TXT 파싱 오류 {file_path}: {e}")
        return []

def parse_md(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [f.read()]
    except Exception as e:
        print(f"MD 파싱 오류 {file_path}: {e}")
        return []

def parse_csv(file_path: str) -> List[str]:
    try:
        df = pd.read_csv(file_path)
        texts = []
        for col in df.columns:
            texts.extend([str(x) for x in df[col].dropna() if str(x).strip()])
        return texts
    except Exception as e:
        print(f"CSV 파싱 오류 {file_path}: {e}")
        return []

def process_guide_documents() -> List[Dict]:
    if not os.path.isdir(GUIDE_DIR):
        print(f"[WARN] GUIDE_DIR not found: {GUIDE_DIR} → 가이드 문서를 스킵합니다.")
        return []
    all_chunks: List[Dict] = []
    for filename in os.listdir(GUIDE_DIR):
        file_path = os.path.join(GUIDE_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXT:
            continue
        print(f"Processing guide file: {filename}")
        if   ext == '.pdf':  texts = parse_pdf(file_path)
        elif ext == '.docx': texts = parse_docx(file_path)
        elif ext in ('.xlsx', '.xls'): texts = parse_excel(file_path)
        elif ext == '.txt':  texts = parse_txt(file_path)
        elif ext == '.md':   texts = parse_md(file_path)
        elif ext == '.csv':  texts = parse_csv(file_path)
        elif ext == '.jsonl':
            docs = parse_law_jsonl(file_path)
            texts = [d.get("text","") for d in docs if d.get("text")]
        else:
            texts = []

        if not texts:
            continue
        metadata = {'source': filename, 'doc_type': 'guide'}
        chunks = make_chunks(texts)
        for i, chunk in enumerate(chunks):
            chunk_md = dict(metadata)
            chunk_md['chunk_id'] = f"{filename}_{i}"
            chunk_md['chunk_index'] = i
            all_chunks.append({'text': chunk, 'metadata': chunk_md})
    return all_chunks

# =========================================
# 임베딩 및 벡터 DB (Chroma)
# =========================================
_embedding_model = None
_chroma_client = None
_collection = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
    return _chroma_client

def _recreate_collection(client):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def ensure_collection_ok():
    """HNSW 손상 등으로 컬렉션 접근 실패 시 재생성하고 False 반환."""
    client = get_chroma_client()
    try:
        col = client.get_collection(COLLECTION_NAME)
        _ = col.count()  # 간단 상태 확인
        return True
    except Exception as e:
        msg = str(e).lower()
        if "hnsw" in msg or "segment reader" in msg or "internalerror" in msg:
            print("[WARN] Chroma HNSW 손상 감지 → 컬렉션 재생성")
            new_col = _recreate_collection(client)
            globals()["_collection"] = new_col
            return False
        raise

def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = _recreate_collection(client)
    return _collection

def embed_documents(chunks: List[Dict]):
    collection = get_collection()
    model = get_embedding_model()

    # 안전 초기화(기존 전체 삭제 실패 시 컬렉션 재생성)
    try:
        collection.delete(where={"doc_type": {"$exists": True}})
    except Exception:
        client = get_chroma_client()
        new_col = _recreate_collection(client)
        globals()["_collection"] = new_col
        collection = new_col

    if not chunks:
        print("[WARN] 임베딩할 청크가 없습니다. (law/guide 모두 비어 있음)")
        return

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"doc_{i}_{j}" for j in range(len(batch))]
        embeddings = model.encode(texts).tolist()
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        total_batches = (len(chunks) + batch_size - 1) // batch_size  # ceil(len/chunk)
        print(f"Processed batch {(i // batch_size) + 1}/{total_batches}")

def search_documents(query: str, top_k: int = 10) -> List[Dict]:
    ok = ensure_collection_ok()
    if not ok:
        # 새 컬렉션은 비어 있으므로 임베딩부터 유도
        raise RuntimeError("컬렉션을 재생성했습니다. 먼저 `python law_rag.py --embed` 를 다시 실행해 주세요.")
    collection = get_collection()
    model = get_embedding_model()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    docs = []
    for i in range(len(results.get('documents', [[]])[0])):
        docs.append({
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i]
        })
    return docs

# =========================================
# (옵션) LLM 답변 생성 - 환경에 GPU/메모리 여건 필요
# =========================================
_llm_model = None

def get_llm_model():
    global _llm_model
    if _llm_model is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tok = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        mdl = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        _llm_model = (tok, mdl)
    return _llm_model

def generate_with_llm(question: str, snippets: List[Dict]) -> str:
    try:
        tok, mdl = get_llm_model()
        context = "\n\n".join([s['text'] for s in snippets[:5]])
        prompt = f"""당신은 금융 보안 전문가 입니다. 다음은 법률 및 가이드 문서의 관련 내용입니다:

{context}

질문: {question}

위의 내용을 바탕으로 정확하고 간결하게 답변해주세요. 객관식 문제인 경우 정답 번호만 답해주세요. 내용을 답할때 차근차근히 생각하면서 답해주세요"""
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        import torch
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True, pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0], skip_special_tokens=True)
        return resp.replace(prompt, "").strip()
    except Exception as e:
        print(f"LLM 생성 오류: {e}")
        return "관련 정보를 찾을 수 없습니다."

def _is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) == 0: return True
    if t.lower() in {"0","none","n/a","null"}: return True
    if re.fullmatch(r"[0\s\.]+", t): return True
    if re.fullmatch(r"[0-9%\s\.,\-\/]+", t) and len(t) < 30: return True
    zeros = t.count("0")
    return zeros >= 5 and zeros / max(1, len(t)) > 0.4

def _fallback_answer(summary: str, snippets: List[Dict], max_len: int = 600) -> str:
    bullets = [s["text"] for s in snippets if s.get("is_keypoint")] or [s["text"] for s in snippets]
    bullets = bullets[:3]
    base = summary if summary and summary != "관련 스니펫을 찾았습니다. 아래를 참고하세요." else ""
    lines = []
    if base: lines.append(base)
    for i, b in enumerate(bullets, 1):
        lines.append(f"{i}. {b}")
    return "\n".join(lines).strip().replace("\n", " ")[:max_len]

def safe_generate_with_llm(question: str, snippets: List[Dict]) -> str:
    try:
        text = generate_with_llm(question, snippets)
        if _is_degenerate(text):
            text = generate_with_llm(question, snippets)
        if _is_degenerate(text):
            return _fallback_answer("", snippets)
        return text
    except Exception:
        return _fallback_answer("", snippets)

# =========================================
# 메인 함수
# =========================================
def embed_all_documents():
    print("법률 문서 처리 중...", flush=True)
    law_chunks = process_law_documents()
    print(f"법률 문서 청크 수: {len(law_chunks)}", flush=True)

    print("가이드 문서 처리 중...", flush=True)
    guide_chunks = process_guide_documents()
    print(f"가이드 문서 청크 수: {len(guide_chunks)}", flush=True)

    all_chunks = law_chunks + guide_chunks
    print(f"전체 청크 수: {len(all_chunks)}", flush=True)

    print("임베딩 중...", flush=True)
    embed_documents(all_chunks)
    print("임베딩 완료!", flush=True)

def ask_question(question: str) -> str:
    print(f"\n=== 질문 처리 시작 ===", flush=True)
    print(f"질문: {question}", flush=True)
    print("문서 검색 중...", flush=True)
    snippets = search_documents(question, top_k=5)
    if not snippets:
        print("관련 문서를 찾을 수 없습니다.", flush=True)
        return "관련 정보를 찾을 수 없습니다."

    print(f"검색된 문서 수: {len(snippets)}", flush=True)
    for i, sn in enumerate(snippets[:3]):
        print(f"문서 {i+1}: {sn['text'][:100]}...", flush=True)

    # 간단 주관식 요약(LLM 비사용 기본)
    relevant = []
    for sn in snippets[:3]:
        sents = sent_split(sn['text'])
        if sents:
            relevant.append(sents[0])
    return " ".join(relevant[:2]) if relevant else snippets[0]['text'][:200]

# =========================================
# CSV/XLSX 배치 유틸 & 처리
# =========================================
def _read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def _load_table(file_path: str) -> pd.DataFrame:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {p}")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    elif p.suffix.lower() == ".csv":
        return _read_csv_safely(p)
    else:
        raise ValueError(f"지원하지 않는 파일 확장자입니다: {p.suffix} (csv/xlsx만 지원)")

def _pick_columns(df: pd.DataFrame):
    cols = [c.strip() for c in df.columns.astype(str)]
    q_candidates = ["Question", "question", "질문", "문항", "Q", "text", "prompt"]
    id_candidates = ["ID", "Id", "id", "번호", "index"]
    q_col = next((c for c in cols if c in q_candidates), None) or cols[0]
    id_col = next((c for c in cols if c in id_candidates), None)
    if id_col is None:
        id_col = "ID"
        df[id_col] = range(1, len(df)+1)
    return q_col, id_col

def _is_hnsw_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("hnsw" in s) or ("segment reader" in s) or ("internalerror" in s)

def process_csv_questions(csv_path: str, output_path: str):
    """CSV/XLSX 배치 처리: 인코딩 자동, 컬럼 자동, 20행마다 자동 저장, 친절 로그"""
    print(f"\n=== CSV 파일 처리 시작 ===", flush=True)
    print(f"입력 파일: {csv_path}", flush=True)

    # 테이블 로드
    try:
        df = _load_table(csv_path)
    except Exception as e:
        print(f"[ERROR] 입력 파일을 읽는 중 오류: {e}", flush=True)
        raise

    print(f"컬럼: {list(df.columns)} / 총 질문 수: {len(df)}", flush=True)
    if len(df) == 0:
        print("[WARN] 입력 파일에 행이 없습니다. 종료합니다.", flush=True)
        return

    # 컬럼 선택
    q_col, id_col = _pick_columns(df)
    print(f"질문 컬럼: {q_col} / ID 컬럼: {id_col}", flush=True)

    # 출력 경로
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"출력 파일: {out_path}", flush=True)

    answers = []
    SAVE_EVERY = 20

    for idx, row in df.iterrows():
        raw_q = row.get(q_col)
        q = "" if pd.isna(raw_q) else str(raw_q).strip()

        print("\n" + "="*50, flush=True)
        print(f"처리 중: {idx+1}/{len(df)}  (ID={row.get(id_col, idx+1)})", flush=True)
        print(f"질문: {q[:120] + ('...' if len(q)>120 else '')}", flush=True)

        if not q:
            print("[WARN] 비어있는 질문 → 빈 답변", flush=True)
            answers.append("")
        else:
            try:
                ans = ask_question(q)
                answers.append(ans)
                print(f"답변 완료: {ans[:160] + ('...' if len(ans)>160 else '')}", flush=True)
            except Exception as e:
                if _is_hnsw_error(e):
                    print("[ERROR] Chroma HNSW 인덱스 손상 감지.", flush=True)
                    print("       해결: PERSIST_DIR를 동기화 폴더 밖으로 지정하고 재임베딩하세요.", flush=True)
                    print("       예)  $env:PERSIST_DIR='C:\\chroma_store' ; python law_rag.py --embed", flush=True)
                else:
                    print(f"[ERROR] 질문 처리 중 예외: {e}", flush=True)
                answers.append("")

        # 중간 저장
        if (idx + 1) % SAVE_EVERY == 0:
            df_out = df.copy()
            df_out["Answer"] = answers + [""] * (len(df_out) - len(answers))
            try:
                df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"[자동 저장] {idx+1}개 처리 → {out_path}", flush=True)
            except Exception as e:
                print(f"[WARN] 중간 저장 실패: {e}", flush=True)

    # 최종 저장
    df["Answer"] = answers
    try:
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n=== 결과 저장 완료: {out_path} ===", flush=True)
    except Exception as e:
        print(f"[ERROR] 최종 저장 실패: {e}", flush=True)
        raise

# =========================================
# CLI
# =========================================
def main():
    parser = argparse.ArgumentParser(description="Law RAG System")
    parser.add_argument("--embed", action="store_true", help="문서 임베딩")
    parser.add_argument("--ask", type=str, help="질문")
    parser.add_argument("--ask-csv", type=str, help="CSV/XLSX 파일의 질문들 처리")
    parser.add_argument("--output", type=str, default="submission.csv", help="출력 파일 경로")
    args = parser.parse_args()

    if args.embed:
        embed_all_documents()
    elif args.ask:
        answer = ask_question(args.ask)
        print(f"질문: {args.ask}")
        print(f"답변: {answer}")
    elif args.ask_csv:
        process_csv_questions(args.ask_csv, args.output)
    else:
        print("사용법: python law_rag.py --embed 또는 python law_rag.py --ask '질문' 또는 python law_rag.py --ask-csv test.csv")

# =========================================
# 간단 조회용 FastAPI 엔드포인트
# =========================================
@app.get("/health")
def health():
    return {"status": "ok", "LAW_DIR": LAW_DIR, "GUIDE_DIR": GUIDE_DIR, "PERSIST_DIR": PERSIST_DIR}

@app.get("/collections")
def list_collections():
    client = get_chroma_client()
    cols = [c.name for c in client.list_collections()]
    return {"collections": cols}

@app.get("/peek")
def peek(collection: str = Query(default=COLLECTION_NAME), n: int = 3):
    col = get_collection() if collection == COLLECTION_NAME else get_chroma_client().get_collection(collection)
    res = col.peek(n)
    return res

@app.get("/search")
def api_search(q: str, k: int = 5, collection: str = Query(default=COLLECTION_NAME)):
    col = get_collection() if collection == COLLECTION_NAME else get_chroma_client().get_collection(collection)
    model = get_embedding_model()
    emb = model.encode([q]).tolist()
    res = col.query(query_embeddings=emb, n_results=k, include=["documents","metadatas","distances"])
    hits = []
    for i, doc in enumerate(res.get("documents", [[]])[0]):
        hits.append({
            "distance": res["distances"][0][i],
            "metadata": res["metadatas"][0][i],
            "document": doc
        })
    return {"query": q, "hits": hits}

if __name__ == "__main__":
    main()