# llm_client.py
import json, time
from typing import Any, List, Tuple
from openai import OpenAI
from settings import settings
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, SUGGEST_PROMPT


def _parse_json_block(text: str) -> dict[str, Any]:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        raise ValueError("LLM 출력에서 JSON을 찾지 못함")
    return json.loads(text[s:e+1])


def _extract_output_text(resp) -> str:
    # Responses API 편의 프로퍼티
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # fallback: 트리에서 text 찾기
    try:
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                if getattr(part, "text", None):
                    return part.text
    except Exception:
        pass
    return str(resp)


def call_llm(title: str, body: str) -> Tuple[str, str, int, int, str, int]:
    """새 제목/요약 생성"""
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


def suggest_questions(title: str, body: str, n: int = 4) -> Tuple[List[str], int, int, str, int]:
    """추천 질문 n개 생성"""
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수를 확인하세요.")

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    prompt = SUGGEST_PROMPT.format(
        title=title, body=body[:settings.MAX_BODY_CHARS], n=4
    )
    t0 = time.time()
    # 타입 경고 피하려고 문자열 하나로 input 전달
    resp = client.responses.create(
        model=settings.MODEL_NAME,
        input=prompt,
        temperature=0.6,
        max_output_tokens=300,
    )
    text = _extract_output_text(resp)
    data = _parse_json_block(text)
    usage = getattr(resp, "usage", None)
    meta_in = getattr(usage, "input_tokens", 0) if usage else 0
    meta_out = getattr(usage, "output_tokens", 0) if usage else 0
    picked_model = getattr(resp, "model", settings.MODEL_NAME)
    latency_ms = int((time.time() - t0) * 1000)

    return list(data.get("questions", [])), meta_in, meta_out, picked_model, latency_ms
