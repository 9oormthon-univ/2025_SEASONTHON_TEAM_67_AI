from fastapi import FastAPI, HTTPException
import asyncio
from schemas import (
    RewriteRequest, RewriteResponse, TokensUsed,
    RewriteBatchRequest, RewriteBatchResponse, RewriteBatchItemResult, RewriteBatchItemIn, ChatArticleRequest, ChatArticleResponse
)
from llm_client import call_llm, suggest_questions_and_quiz, chat_about_article

app = FastAPI(title="Article Rewriter API", version="1.3.1")

CONCURRENCY_ARTICLES = 4 #한번에 요청 개수 제한

@app.post("/v1/rewrite-summarize", response_model=RewriteResponse)
async def rewrite_summarize(payload: RewriteRequest):
    body = payload.body.strip()
    if len(body) < 50:
        raise HTTPException(status_code=422, detail="본문이 너무 짧습니다(최소 50자).")
    try:
        new_title, summary, in1, out1, model, lat1 = call_llm(payload.title, body)
        questions, quiz, in2, out2, _, lat2 = suggest_questions_and_quiz(payload.title, body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RewriteResponse(
        articleId=payload.articleId,
        newTitle=new_title.strip(),
        summary=summary.strip(),
        questions=questions,
        quiz=quiz,
        tokensUsed=TokensUsed(input=in1 + in2, output=out1 + out2),
        model=model,
        latencyMs=lat1 + lat2
    )

@app.post("/v1/rewrite-batch", response_model=RewriteBatchResponse)
async def rewrite_batch(payload: RewriteBatchRequest):
    sem = asyncio.Semaphore(CONCURRENCY_ARTICLES)

    async def process_one(item: RewriteBatchItemIn) -> RewriteBatchItemResult:
        async with sem:
            body = (item.body or "").strip()
            if len(body) < 50:
                return RewriteBatchItemResult(
                    articleId=item.articleId, ok=False,
                    error="본문이 너무 짧습니다(최소 50자)."
                )
            try:
                new_title, summary, in1, out1, model, lat1 = await asyncio.to_thread(
                    call_llm, item.title, body
                )
                questions, quiz, in2, out2, _, lat2 = await asyncio.to_thread(
                    suggest_questions_and_quiz, item.title, body
                )
                resp = RewriteResponse(
                    articleId=item.articleId,
                    newTitle=new_title.strip(),
                    summary=summary.strip(),
                    questions=questions,
                    quiz=quiz,
                    tokensUsed=TokensUsed(input=in1 + in2, output=out1 + out2),
                    model=model,
                    latencyMs=lat1 + lat2
                )
                return RewriteBatchItemResult(articleId=item.articleId, ok=True, data=resp)
            except Exception as e:
                return RewriteBatchItemResult(articleId=item.articleId, ok=False, error=str(e))

    tasks = [process_one(it) for it in payload.items]
    results = await asyncio.gather(*tasks)
    return RewriteBatchResponse(results=list(results))


@app.post("/v1/chat-article", response_model=ChatArticleResponse)
async def chat_article(payload: ChatArticleRequest):
    try:
        answer, model, latency = chat_about_article(
            payload.articleId,
            payload.userId,
            payload.summary,
            payload.history,
            payload.userMessage,
        )
        return ChatArticleResponse(
            articleId=payload.articleId,
            userId=payload.userId,
            userMessage=payload.userMessage,
            answer=answer,
            model=model,
            latencyMs=latency,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}