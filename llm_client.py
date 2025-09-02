import json, time
from typing import Any
from openai import OpenAI
from settings import settings
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

def _parse_json_block(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 출력에서 JSON을 찾지 못함")
    return json.loads(text[start:end+1])

def _extract_output_text(resp) -> str:
    # 새 SDK에 있는 편의 프로퍼티
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # output 트리를 탐색
    try:
        for item in getattr(resp, "output", []) or []:
            content = getattr(item, "content", []) or []
            for part in content:
                if getattr(part, "type", None) == "output_text" and getattr(part, "text", None):
                    return part.text
                if hasattr(part, "text"):
                    return part.text
    except Exception:
        pass
    # 문자열로 변환
    return str(resp)

def call_llm(title: str, body: str):
    """LLM 호출해서 새 제목/요약 생성"""
    if not settings.OPENAI_API_KEY:
        print("OPENAI_API_KEY가 비어 있습니다. .env 또는 환경변수를 확인하세요.")

    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, body=body[:settings.MAX_BODY_CHARS])
    t0 = time.time()

    client = OpenAI(api_key=settings.OPENAI_API_KEY)


    #client = OpenAI()  # OPENAI_API_KEY는 환경변수에서 자동 인식


    resp = client.responses.create(
        model=settings.MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_output_tokens=400
    )

    text = _extract_output_text(resp)
    usage = getattr(resp, "usage", None)
    meta_in = getattr(usage, "input_tokens", 0) if usage else 0
    meta_out = getattr(usage, "output_tokens", 0) if usage else 0
    picked_model = getattr(resp, "model", settings.MODEL_NAME)
    latency_ms = int((time.time() - t0) * 1000)

    parsed = _parse_json_block(text)
    return parsed["newTitle"], parsed["summary"], meta_in, meta_out, picked_model, latency_ms
