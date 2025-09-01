# 📚 Law RAG – 법률 문서 기반 RAG 시스템

> **법률·가이드 문서 자동 분석 및 질의응답 시스템**  
> Retrieval-Augmented Generation(RAG) 기법을 활용하여 방대한 법률 문서를 효율적으로 검색·요약하고, 질의응답과 배치 처리를 지원합니다.

---

## 🎯 프로젝트 개요

이 프로젝트는 **데이콘 AI 금융 챌린지**에서 제시된 과제를 해결하기 위해 개발되었습니다.  
금융·법률 문서는 방대하고 복잡하여, **실무자들이 필요한 조항을 신속히 찾기 어렵다**는 문제가 있습니다.  
이를 해결하기 위해 본 프로젝트는 다음과 같은 목표로 설계되었습니다.

1. **문서 자동화** – 다양한 형식(PDF, DOCX, XLSX, CSV 등)의 법률/가이드 문서를 자동 파싱  
2. **효율적 검색** – 벡터 DB(Chroma) + SentenceTransformer를 통한 의미 기반 검색  
3. **실시간 질의응답** – FastAPI 서버를 통한 검색/요약 API 제공  
4. **배치 처리** – CSV/XLSX 파일의 대량 질문에 대해 자동 응답 생성  

---

## 🛠 기술 스택

- **언어/프레임워크**: Python, FastAPI, Uvicorn  
- **AI/ML**: SentenceTransformers(`jhgan/ko-sroberta-multitask`), HuggingFace Transformers (`gemma-ko-7b`)  
- **Vector DB**: Chroma (Persistent Client)  
- **문서 파서**: pdfplumber, python-docx, openpyxl, pandas  
- **배포 고려**: Docker + REST API 확장 가능  

---

## 🚀 주요 기능

- 📂 **문서 파싱 및 청크화**  
  - `rag_laws/` : JSONL 법률 조항  
  - `guide/` : PDF, DOCX, XLSX, TXT, CSV 등  
  - 자동 문장 분리 + overlap 기반 청크 생성  

- 🔎 **검색 및 요약 응답**  
  - 의미 기반 Top-K 검색  
  - 간단 요약형 응답 기본, LLM 기반 답변 선택 가능  

- 🧾 **CSV/XLSX 배치 처리**  
  - `--ask-csv` 옵션으로 다수 질문 처리  
  - 인코딩 자동 판별, 컬럼 자동 인식, 20행마다 중간 저장  

- 🌐 **API 제공**  
  - `/health` : 상태 확인  
  - `/search` : 의미 기반 검색  
  - `/collections`, `/peek` : DB 상태 확인  

- 🛡 **안정성**  
  - HNSW 인덱스 손상 자동 감지 → 컬렉션 재생성  
  - 경로 안정화(BASE_DIR 기준)  
  - 에러 발생 시 친절한 로그 출력  

---

## ⚡ 실행 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt

### 2. 문서 임베딩
```bash
python law_rag.py --embed
```

### 3. 단일 질의
```bash
python law_rag.py --ask "개인정보 처리 위반 시 처벌 규정은?"
```

### 4. CSV/XLSX 배치 처리
```bash
python law_rag.py --ask-csv ./questions.csv --output ./submission.csv
```

### 5. FastAPI 서버 실행
```bash
uvicorn law_rag:app --reload
```

---

## 📂 프로젝트 구조
```
law_rag.py        # 메인 RAG 시스템
rag_laws/         # 법률 JSONL 파일
guide/            # 가이드 문서
chroma_store/     # 벡터DB 저장소
```

---

## 💡 차별점 & 학습 포인트

- 실제 금융 챌린지 문제 해결을 목표로 개발  
- 키워드 매칭이 아닌 의미 기반 검색으로 맥락 이해 가능  
- 법률 문서 특유의 긴 문장도 청크 단위 처리  
- HNSW 인덱스 손상 자동 복구, 환경 변수 기반 경로 안정화  
- CSV/XLSX 배치 처리로 시험 문제/업무 질의에 즉시 활용 가능  

---

## 📌 사용 예시
```bash
# 문서 임베딩 후 질의
python law_rag.py --embed
python law_rag.py --ask "전자금융거래에서 인증 관련 의무사항은?"

# CSV 질문 일괄 처리
python law_rag.py --ask-csv ./questions.xlsx --output ./answers.csv
```

---

## 👤 프로젝트 동기

이 프로젝트는 **데이콘 AI 금융 챌린지** 참가를 계기로 시작했습니다.  
실제 금융 규제 및 법률 관련 문제를 해결하기 위해 **데이터 전처리, 의미 기반 검색, 대규모 배치 처리**를 경험하며  
실무 친화적인 AI 시스템을 설계할 수 있었습니다.
