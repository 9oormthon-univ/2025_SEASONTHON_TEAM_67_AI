from fastapi import FastAPI, HTTPException
import asyncio
from schemas import (
    RewriteRequest, RewriteResponse, TokensUsed,
    RewriteBatchRequest, RewriteBatchResponse, RewriteBatchItemResult, RewriteBatchItemIn, ChatArticleRequest, ChatArticleResponse,
    RewriteBatchMultiResponse, RewriteBatchItemMultiResult,RewriteVariant, RewriteMultiResponse
)
from llm_client import call_llm, suggest_questions_and_quiz, chat_about_article, evaluate_epi, run_one_style

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
        epi_json, in3, out3, _, lat3, reason = evaluate_epi(payload.title, body, new_title, summary)

        epi = {
            "epiOriginal": int(epi_json["original"]["EPI"]),
            "epiSummary": int(epi_json["summary"]["EPI"]),
            "reductionPct": float(epi_json.get("reductionPct", 0)),
            "stimulationReduced": str(epi_json.get("stimulationReduced", "자극도를 0% 줄였어요")),
            "componentsOriginal": {k: float(epi_json["original"][k]) for k in
                                   ("S","SUBJ","K","F","C","V","X","EVID")},
            "componentsSummary":  {k: float(epi_json["summary"][k])  for k in
                                   ("S","SUBJ","K","F","C","V","X","EVID")},
            "reason": reason
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return RewriteResponse(
        articleId=payload.articleId,
        newTitle=new_title.strip(),
        summary=summary.strip(),
        questions=questions,
        quiz=quiz,
        tokensUsed=TokensUsed(input=in1 + in2 + in3, output=out1 + out2 + out3),
        model=model,
        latencyMs=lat1 + lat2 + lat3,
        epi=epi
    )

@app.post("/v1/rewrite-summarize3", response_model=RewriteMultiResponse)
async def rewrite_summarize3(payload: RewriteRequest):
    body = payload.body.strip()
    if len(body) < 50:
        # FastAPI의 HTTPException 대신, 명확한 422 메시지 유지
        raise HTTPException(status_code=422, detail="본문이 너무 짧습니다(최소 50자).")

    styles = ["CONCISE", "FRIENDLY", "NEUTRAL"]

    async def run(style):
        return await asyncio.to_thread(run_one_style, style, payload.title, body)

    # 3개 스타일 병렬 실행
    results = await asyncio.gather(*(run(s) for s in styles))

    # totals
    total_in = sum(r["tokensUsed"]["input"] for r in results)
    total_out = sum(r["tokensUsed"]["output"] for r in results)
    total_latency = sum(r["latencyMs"] for r in results)

    variants = []
    for r in results:
        variants.append(RewriteVariant(
            newsStyle=r["newsStyle"],
            articleId=payload.articleId,
            newTitle=r["newTitle"],
            summary=r["summary"],
            questions=r["questions"],
            quiz=r["quiz"],
            tokensUsed=TokensUsed(**r["tokensUsed"]),
            model=r["model"],
            latencyMs=r["latencyMs"],
            epi=r["epi"]
        ))

    return RewriteMultiResponse(
        articleId=payload.articleId,
        variants=variants,
        tokensUsedTotal=TokensUsed(input=total_in, output=total_out),
        latencyMsTotal=total_latency
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
                epi_json, in3, out3, _, lat3, reason = await asyncio.to_thread(
                    evaluate_epi, item.title, body, new_title, summary
                )

                epi = {
                    "epiOriginal": int(epi_json["original"]["EPI"]),
                    "epiSummary": int(epi_json["summary"]["EPI"]),
                    "reductionPct": float(epi_json.get("reductionPct", 0)),
                    "stimulationReduced": str(epi_json.get("stimulationReduced", "자극도를 0% 줄였어요")),
                    "componentsOriginal": {k: float(epi_json["original"][k]) for k in
                                           ("S", "SUBJ", "K", "F", "C", "V", "X", "EVID")},
                    "componentsSummary": {k: float(epi_json["summary"][k]) for k in
                                          ("S", "SUBJ", "K", "F", "C", "V", "X", "EVID")},
                    "reason": reason
                }

                resp = RewriteResponse(
                    articleId=item.articleId,
                    newTitle=new_title.strip(),
                    summary=summary.strip(),
                    questions=questions,
                    quiz=quiz,
                    tokensUsed=TokensUsed(input=in1 + in2 + in3, output=out1 + out2 + out3),
                    model=model,
                    latencyMs=lat1 + lat2 + lat3,
                    epi=epi
                )
                return RewriteBatchItemResult(articleId=item.articleId, ok=True, data=resp)
            except Exception as e:
                return RewriteBatchItemResult(articleId=item.articleId, ok=False, error=str(e))

    tasks = [process_one(it) for it in payload.items]
    results = await asyncio.gather(*tasks)
    return RewriteBatchResponse(results=list(results))

CONCURRENCY_ARTICLES = 4

@app.post("/v1/rewrite-batch3", response_model=RewriteBatchMultiResponse)
async def rewrite_batch3(payload: RewriteBatchRequest):
    sem = asyncio.Semaphore(CONCURRENCY_ARTICLES)

    async def process_one(item) -> RewriteBatchItemMultiResult:
        async with sem:
            body = (item.body or "").strip()
            if len(body) < 50:
                return RewriteBatchItemMultiResult(
                    articleId=item.articleId, ok=False,
                    error="본문이 너무 짧습니다(최소 50자)."
                )
            try:
                styles = ["CONCISE", "FRIENDLY", "NEUTRAL"]

                async def run(style):
                    return await asyncio.to_thread(run_one_style, style, item.title, body)

                results = await asyncio.gather(*(run(s) for s in styles))
                total_in = sum(r["tokensUsed"]["input"] for r in results)
                total_out = sum(r["tokensUsed"]["output"] for r in results)
                total_latency = sum(r["latencyMs"] for r in results)

                variants = []
                for r in results:
                    variants.append(RewriteVariant(
                        newsStyle=r["newsStyle"],
                        articleId=item.articleId,
                        newTitle=r["newTitle"],
                        summary=r["summary"],
                        questions=r["questions"],
                        quiz=r["quiz"],
                        tokensUsed=TokensUsed(**r["tokensUsed"]),
                        model=r["model"],
                        latencyMs=r["latencyMs"],
                        epi=r["epi"]
                    ))

                data = RewriteMultiResponse(
                    articleId=item.articleId,
                    variants=variants,
                    tokensUsedTotal=TokensUsed(input=total_in, output=total_out),
                    latencyMsTotal=total_latency
                )
                return RewriteBatchItemMultiResult(articleId=item.articleId, ok=True, data=data)
            except Exception as e:
                return RewriteBatchItemMultiResult(articleId=item.articleId, ok=False, error=str(e))

    tasks = [process_one(it) for it in payload.items]
    results = await asyncio.gather(*tasks)
    return RewriteBatchMultiResponse(results=list(results))

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
    return {"status": "재 배포 성공!!"}