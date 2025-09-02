import json, time
from typing import Any, List, Tuple, Dict
from openai import OpenAI
from settings import settings
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, SUGGEST_WITH_QUIZ_PROMPT, CHAT_SYSTEM_PROMPT

def _parse_json_block(text: str) -> dict[str, Any]:
    """
    LLM 출력에서 첫 '{'와 마지막 '}' 사이를 잘라 JSON 파싱.
    실패 시 어떤 내용 때문에 깨졌는지 앞부분을 detail에 담아 디버깅하기 쉽게 만든다.
    """
    # 혹시 코드블록(```json ... ```)이 온 경우를 대비해 간단히 제거
    if "```" in text:
        parts = text.split("```")
        # 코드블록 내부를 우선 사용 (예: ```json {..} ```)
        for p in parts:
            if "{" in p and "}" in p:
                text = p
                break

    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        snippet = text.strip().replace("\n", "\\n")
        raise ValueError(f"LLM 출력에서 JSON을 찾지 못함: '{snippet[:180]}'")

    raw = text[s:e+1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as ex:
        # 어디서 깨졌는지 보여주기 위해 원본 일부를 detail에 포함
        preview = raw[:400].replace("\n", "\\n")
        raise ValueError(f"JSON 파싱 실패: {ex}. 원본 일부: '{preview}'")

def _extract_output_text(resp) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    try:
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                if getattr(part, "text", None):
                    return part.text
    except Exception:
        pass
    return str(resp)

def _normalize_yes_no(val: str) -> str:
    v = (val or "").strip().upper()
    if v in ("YES", "Y"):
        return "YES"
    if v in ("NO", "N"):
        return "NO"
    raise ValueError(f"quiz.answer가 YES/NO가 아님: {val!r}")

def call_llm(title: str, body: str) -> Tuple[str, str, int, int, str, int]:
    """호출#1: 새 제목/요약"""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수를 확인하세요.")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=title, body=body[:settings.MAX_BODY_CHARS]
    )
    t0 = time.time()
    resp = client.responses.create(
        model=settings.MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_output_tokens=400,
    )
    text = _extract_output_text(resp)
    usage = getattr(resp, "usage", None)
    meta_in = getattr(usage, "input_tokens", 0) if usage else 0
    meta_out = getattr(usage, "output_tokens", 0) if usage else 0
    picked_model = getattr(resp, "model", settings.MODEL_NAME)
    latency_ms = int((time.time() - t0) * 1000)

    parsed = _parse_json_block(text)
    return parsed["newTitle"], parsed["summary"], meta_in, meta_out, picked_model, latency_ms

def suggest_questions_and_quiz(title: str, body: str) -> Tuple[List[str], Dict[str, str], int, int, str, int]:
    """호출#2: 질문 4개 + 예/아니오 퀴즈 1개(정답 YES/NO)"""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수를 확인하세요.")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    prompt = SUGGEST_WITH_QUIZ_PROMPT.format(
        title=title, body=body[:settings.MAX_BODY_CHARS]
    )
    t0 = time.time()
    resp = client.responses.create(
        model=settings.MODEL_NAME,
        input=prompt,                  # 문자열 하나로 전달
        temperature=0.6,
        max_output_tokens=320,
    )
    text = _extract_output_text(resp)
    data = _parse_json_block(text)

    # 후처리: questions 4개 보장, quiz.answer 정상화
    questions = data.get("questions", [])
    if not isinstance(questions, list):
        questions = []
    questions = [q for q in questions if isinstance(q, str)][:4]

    quiz = data.get("quiz", {})
    if not isinstance(quiz, dict) or "question" not in quiz or "answer" not in quiz:
        raise ValueError("quiz 필드가 없거나 형식이 올바르지 않습니다.")
    quiz = {
        "question": str(quiz.get("question", "")).strip(),
        "answer": _normalize_yes_no(str(quiz.get("answer", "")).strip())
    }

    usage = getattr(resp, "usage", None)
    meta_in  = getattr(usage, "input_tokens", 0) if usage else 0
    meta_out = getattr(usage, "output_tokens", 0) if usage else 0
    picked_model = getattr(resp, "model", settings.MODEL_NAME)
    latency_ms = int((time.time() - t0) * 1000)

    return questions, quiz, meta_in, meta_out, picked_model, latency_ms

def chat_about_article(article_id: str, user_id: str, summary: str, history: list, user_msg: str):
    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "system", "content": f"[기사ID: {article_id}]"},
        {"role": "system", "content": f"[기사 요약]\n{summary}"}
    ]

    for h in history:
        messages.append({"role": h.role, "content": h.content})

    messages.append({"role": "user", "content": user_msg})

    t0 = time.time()
    resp = client.chat.completions.create(
        model=settings.MODEL_NAME,
        messages=messages,
        temperature=0.5,
        max_tokens=400,
    )
    latency = int((time.time() - t0) * 1000)
    answer = resp.choices[0].message.content
    model_used = resp.model

    return answer, model_used, latency