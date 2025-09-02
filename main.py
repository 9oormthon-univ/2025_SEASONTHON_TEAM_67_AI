from fastapi import FastAPI, HTTPException
from schemas import RewriteRequest, RewriteResponse, TokensUsed
from settings import settings
from llm_client import call_llm

app = FastAPI(title="Article Rewriter API", version="1.0.0")

@app.post("/v1/rewrite-summarize", response_model=RewriteResponse)
async def rewrite_summarize(payload: RewriteRequest):
    if len(payload.body.strip()) < 50:
        raise HTTPException(status_code=422, detail="본문이 너무 짧습니다(최소 50자).")
    try:
        new_title, summary, in_tok, out_tok, model, latency = call_llm(payload.title, payload.body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RewriteResponse(
        articleId=payload.articleId,
        newTitle=new_title.strip(),
        summary=summary.strip(),
        tokensUsed=TokensUsed(input=in_tok, output=out_tok),
        model=model,
        latencyMs=latency
    )