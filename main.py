from fastapi import FastAPI, HTTPException
from schemas import RewriteRequest, RewriteResponse, TokensUsed
from llm_client import call_llm, suggest_questions

app = FastAPI(title="Article Rewriter API", version="1.1.0")

@app.post("/v1/rewrite-summarize", response_model=RewriteResponse)
async def rewrite_summarize(payload: RewriteRequest):
    body = payload.body.strip()
    if len(body) < 50:
        raise HTTPException(status_code=422, detail="본문이 너무 짧습니다(최소 50자).")

    try:
        # 요약/새 제목
        new_title, summary, in_tok1, out_tok1, model, latency1 = call_llm(payload.title, body)
        # 추천 질문
        questions, in_tok2, out_tok2, _, latency2 = suggest_questions(payload.title, body, n=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RewriteResponse(
        articleId=payload.articleId,
        newTitle=new_title.strip(),
        summary=summary.strip(),
        questions=questions[:4],
        tokensUsed=TokensUsed(input=in_tok1 + in_tok2, output=out_tok1 + out_tok2),
        model=model,
        latencyMs=latency1 + latency2
    )
