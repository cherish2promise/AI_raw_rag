
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# -----------------------
# 공통: 유틸
# -----------------------
def read_textfile(path: str, encoding="utf-8") -> str:
    return Path(path).read_text(encoding=encoding)

def write_jsonl(records: List[Dict], out_path: str):
    with Path(out_path).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ============================================================
# 파트 A) 법령 파서 + Instruction 생성
# ============================================================
CIRCLED_DIGITS = {
    "①":"1","②":"2","③":"3","④":"4","⑤":"5","⑥":"6","⑦":"7","⑧":"8","⑨":"9","⑩":"10",
    "⑪":"11","⑫":"12","⑬":"13","⑭":"14","⑮":"15","⑯":"16","⑰":"17","⑱":"18","⑲":"19","⑳":"20"
}

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    for k, v in CIRCLED_DIGITS.items():
        s = s.replace(k, f"({v})")  # ① -> (1)
    s = re.sub(r"\u00A0", " ", s)  # NBSP
    s = re.sub(r"\s+", " ", s).strip()
    return s

ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조\s*(?:\(([^)]*)\))?", re.UNICODE)

@dataclass
class Item:
    item_no: Optional[str]
    text: str

@dataclass
class Paragraph:
    paragraph_no: Optional[str]
    text: str
    items: List[Item]

@dataclass
class Article:
    article_no: str
    title: Optional[str]
    text: str
    paragraphs: List[Paragraph]

def split_articles(text: str) -> List[Tuple[str, Optional[str], str]]:
    parts = []
    for m in ARTICLE_RE.finditer(text):
        parts.append((m.start(), m.end(), m.group(1), m.group(2)))
    articles = []
    for i, (s, e, a_no, a_title) in enumerate(parts):
        end = parts[i+1][0] if i+1 < len(parts) else len(text)
        body = text[e:end].strip()
        articles.append((a_no, a_title, body))
    return articles

def split_items(paragraph_text: str) -> List[Item]:
    t = paragraph_text
    # 호/목 경계 (1. | (1) | 가.)
    t = re.sub(r"(?:(?<=\s)|^)([0-9]{1,2})\.(?!\d)", r"\n@@I\1. ", t)
    t = re.sub(r"(?:(?<=\s)|^)(\([0-9]{1,2}\))", r"\n@@I\1 ", t)
    t = re.sub(r"(?:(?<=\s)|^)([가-힣])\.(?=\s)", r"\n@@I\1. ", t)
    chunks = [c.strip() for c in t.split("\n@@I") if c.strip()]
    if len(chunks) == 1:
        return []
    items: List[Item] = []
    for ch in chunks:
        m = re.match(r"^(\(([0-9]{1,2})\)|([0-9]{1,2})\.|([가-힣])\.)\s*", ch)
        if m:
            ino = m.group(2) or m.group(3) or m.group(4)
            body = ch[m.end():].strip()
        else:
            ino, body = None, ch
        items.append(Item(item_no=ino, text=body))
    return items

def split_paragraphs(article_body: str) -> List[Paragraph]:
    tmp = article_body
    # 항 경계 (1) 또는 1.
    tmp = re.sub(r"(?:(?<=\s)|^)(\(\d+\))", r"\n@@P\1 ", tmp)
    tmp = re.sub(r"(?:(?<=\s)|^)(\d{1,2})\.(?!\d)", r"\n@@P(\1) ", tmp)
    chunks = [c.strip() for c in tmp.split("\n@@P") if c.strip()]
    if len(chunks) == 1:
        return [Paragraph(paragraph_no=None, text=chunks[0], items=split_items(chunks[0]))]
    paras: List[Paragraph] = []
    for ch in chunks:
        m = re.match(r"\((\d+)\)\s*", ch)
        pno = m.group(1) if m else None
        body = ch[m.end():].strip() if m else ch
        paras.append(Paragraph(paragraph_no=pno, text=body, items=split_items(body)))
    return paras

def parse_law_text(raw_text: str) -> List[Article]:
    raw = normalize_text(raw_text)
    arts_meta = split_articles(raw)
    articles: List[Article] = []
    for a_no, a_title, a_body in arts_meta:
        paras = split_paragraphs(a_body)
        articles.append(Article(article_no=a_no, title=a_title, text=a_body, paragraphs=paras))
    return articles

def build_records(
    articles: List[Article],
    *,
    law_title: str,
    source_url: str,
    effective_date: Optional[str],
    mode: str = "empty"
) -> List[Dict]:
    """
    mode='empty'        : 요약/설명형 output 비움(이후 LLM 생성/검수용)
    mode='deterministic': 인용/추출형 output을 원문으로 채움(확정형)
    """
    records: List[Dict] = []

    def ref_meta(a_no: str, p_no: Optional[str] = None, i_no: Optional[str] = None):
        return {
            "law_title": law_title,
            "article": int(a_no),
            "paragraph": int(p_no) if p_no else None,
            "item": i_no,
            "url": source_url,
            "effective_date": effective_date,
        }

    for art in articles:
        # 조문 단위
        for ins, ctx in [
            (f"{law_title} 제{art.article_no}조의 주요 내용을 두 문장으로 요약하라.", art.text),
            (f"{law_title} 제{art.article_no}조의 의무 주체와 의무 내용을 식별하라.", art.text),
        ]:
            records.append({
                "instruction": ins,
                "input": ctx,
                "output": "" if mode == "empty" else art.text,
                "references": [ref_meta(art.article_no)],
                "meta": {"level": "article", "task": "summary|extraction"}
            })

        # 항/호 단위
        for para in art.paragraphs:
            pno = para.paragraph_no
            if pno:
                for ins, ctx in [
                    (f"{law_title} 제{art.article_no}조 {pno}항의 요지를 한 문장으로 요약하라.", para.text),
                    (f"{law_title} 제{art.article_no}조 {pno}항에서 규정하는 조건이나 예외를 모두 나열하라.", para.text),
                ]:
                    records.append({
                        "instruction": ins,
                        "input": ctx,
                        "output": "" if mode == "empty" else para.text,
                        "references": [ref_meta(art.article_no, pno)],
                        "meta": {"level": "paragraph", "task": "summary|list"}
                    })
            for it in para.items:
                records.append({
                    "instruction": f"{law_title} 제{art.article_no}조 {pno}항 {it.item_no}호의 내용을 인용하라.",
                    "input": "",
                    "output": it.text if mode == "deterministic" else "",
                    "references": [ref_meta(art.article_no, pno, it.item_no)],
                    "meta": {"level": "item", "task": "quote"}
                })
    return records

def run_task_dataset(args):
    raw = read_textfile(args.input)
    articles = parse_law_text(raw)
    recs = build_records(
        articles,
        law_title=args.law_title,
        source_url=args.source_url,
        effective_date=args.effective_date,
        mode=args.mode,
    )
    write_jsonl(recs, args.output)
    print(f"[OK] {len(recs)} records written to {args.output}")

# ============================================================
# 파트 B) 질문 자동 프롬프트 + LLM 추론 + 정답 추출
# ============================================================
def is_multiple_choice(question_text: str) -> bool:
    """
    객관식 여부 판단: 줄 단위로 '숫자 선택지'가 2개 이상이면 객관식으로 간주
    예) '1 ...', '2 ...' 형태
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str) -> Tuple[str, List[str]]:
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트 분리
    """
    lines = full_text.strip().split("\n")
    q_lines, options = [], []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    question = " ".join(q_lines)
    return question, options

def make_prompt_auto(text: str) -> str:
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )
    return prompt

def extract_answer_only(generated_text: str, original_question: str) -> str:
    """
    - "답변:" 이후 텍스트만 추출
    - 객관식: 정답 숫자만 추출 (실패 시 '0')
    - 주관식: 텍스트 그대로 반환
    - 빈 문자열 방지: "미응답"
    """
    if "답변:" in generated_text:
        text = generated_text.split("답변:")[-1].strip()
    else:
        text = generated_text.strip()

    if not text:
        return "미응답"

    if is_multiple_choice(original_question):
        m = re.match(r"\D*([1-9][0-9]?)", text)
        return m.group(1) if m else "0"
    else:
        return text

def load_generator(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return pipe

def run_task_infer(args):
    # 입력 획득: 단일 파일 or CSV('text' 컬럼)
    questions: List[str] = []
    if args.question_file:
        q = read_textfile(args.question_file)
        questions = [q.strip()]
    elif args.question_csv:
        import pandas as pd
        df = pd.read_csv(args.question_csv)
        if "text" not in df.columns:
            raise ValueError("CSV에는 'text' 컬럼이 있어야 합니다.")
        questions = df["text"].astype(str).tolist()
    else:
        raise ValueError("질문 입력이 필요합니다. --question_file 또는 --question_csv 를 지정하세요.")

    pipe = load_generator(args.model_name)

    preds = []
    for q in questions:
        prompt = make_prompt_auto(q)
        out = pipe(
            prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=pipe.tokenizer.eos_token_id
        )[0]["generated_text"]
        ans = extract_answer_only(out, q)
        preds.append(ans)

    # 결과 출력
    if args.pred_output:
        import pandas as pd
        pd.DataFrame({"text": questions, "pred": preds}).to_csv(args.pred_output, index=False, encoding="utf-8-sig")
        print(f"[OK] {len(preds)} predictions saved to {args.pred_output}")
    else:
        # 단일 질문이면 정답만 출력
        if len(preds) == 1:
            print(preds[0])
        else:
            for i, a in enumerate(preds, 1):
                print(f"{i}\t{a}")

# ============================================================
# 메인: 인자 파싱
# ============================================================
def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["dataset", "infer"], help="실행할 작업 선택")

    # dataset 옵션
    ap.add_argument("--input", help="법령 원문 텍스트(.txt)")
    ap.add_argument("--law_title", help="법령명")
    ap.add_argument("--source_url", help="원문 URL")
    ap.add_argument("--effective_date", default=None, help="시행일(YYYY-MM-DD)")
    ap.add_argument("--output", help="JSONL 출력 경로")
    ap.add_argument("--mode", choices=["empty", "deterministic"], default="empty")

    # infer 옵션
    ap.add_argument("--question_file", help="단일 질문 텍스트 파일 경로")
    ap.add_argument("--question_csv", help="여러 질문 CSV 경로 (컬럼명: text)")
    ap.add_argument("--model_name", default="beomi/gemma-ko-7b", help="허용된 로컬 모델 이름")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--pred_output", help="CSV 저장 경로(옵션). 지정하면 (text, pred)로 저장")
    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.task == "dataset":
        required = ["input", "law_title", "source_url", "output"]
        missing = [x for x in required if getattr(args, x) in (None, "")]
        if missing:
            raise ValueError(f"--task dataset에 필요한 인자 누락: {missing}")
        run_task_dataset(args)

    elif args.task == "infer":
        if not (args.question_file or args.question_csv):
            raise ValueError("--task infer에는 --question_file 또는 --question_csv 중 하나가 필요합니다.")
        run_task_infer(args)

if __name__ == "__main__":
    main()

