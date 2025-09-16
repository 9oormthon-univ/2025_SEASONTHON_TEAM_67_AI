from fastapi import FastAPI, HTTPException
import asyncio
from schemas import (
    RewriteRequest,TokensUsed,
    RewriteBatchRequest,ChatArticleRequest, ChatArticleResponse,
    RewriteBatchMultiResponse, RewriteBatchItemMultiResult,
    RewriteVariant, RewriteMultiResponse
)
from llm_client import suggest_questions_and_quiz, chat_about_article, build_variants_for_styles

app = FastAPI(title="Article Rewriter API", version="1.3.1")

CONCURRENCY_ARTICLES = 4 #한번에 요청 개수 제한

@app.post("/v1/rewrite-summarize3", response_model=RewriteMultiResponse)
async def rewrite_summarize3(payload: RewriteRequest):
    body = payload.body.strip()
    if len(body) < 50:
        raise HTTPException(status_code=422, detail="본문이 너무 짧습니다(최소 50자).")

    styles = ["CONCISE", "FRIENDLY", "NEUTRAL"]

    # [변경] 스타일별 요약/EPI 한번에 생성 (질문/퀴즈는 여기서 만들지 않음)
    variants, v_in, v_out, v_lat = await asyncio.to_thread(
        build_variants_for_styles, payload.title, body, styles
    )

    # [추가] 기사당 1회만 질문/퀴즈 생성
    questions, quiz, q_in, q_out, _, q_lat = await asyncio.to_thread(
        suggest_questions_and_quiz, payload.title, body
    )

    # [변경] per-variant tokens 제거 → 최상위 합계만
    return RewriteMultiResponse(
        articleId=payload.articleId,
        variants=[
            RewriteVariant(
                newsStyle=v["newsStyle"],
                articleId=payload.articleId,
                newTitle=v["newTitle"],
                summary=v["summary"],
                # [삭제] variants 내부 questions/quiz 삭제
                # [삭제] tokensUsed 삭제
                model=v["model"],
                latencyMs=v["latencyMs"],
                epi=v["epi"]
            )
            for v in variants
        ],
        # [추가] 공통 질문/퀴즈를 최상위에 포함시켜야 하므로 스키마에 반영 필요(아래 schemas.py 참고)
        questions=questions,
        quiz=quiz,
        tokensUsedTotal=TokensUsed(input=v_in + q_in, output=v_out + q_out),
        latencyMsTotal=v_lat + q_lat
    )

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

                # [변경] 스타일별 요약/EPI 일괄 생성
                variants, v_in, v_out, v_lat = await asyncio.to_thread(
                    build_variants_for_styles, item.title, body, styles
                )
                # [추가] 기사당 1회만 질문/퀴즈 생성
                questions, quiz, q_in, q_out, _, q_lat = await asyncio.to_thread(
                    suggest_questions_and_quiz, item.title, body
                )

                data = RewriteMultiResponse(
                    articleId=item.articleId,
                    variants=[
                        RewriteVariant(
                            newsStyle=v["newsStyle"],
                            articleId=item.articleId,
                            newTitle=v["newTitle"],
                            summary=v["summary"],
                            # [삭제] variants 내부 questions/quiz/tokensUsed 제거
                            model=v["model"],
                            latencyMs=v["latencyMs"],
                            epi=v["epi"]
                        )
                        for v in variants
                    ],
                    # [추가] 공통 질문/퀴즈: 최상위에 배치
                    questions=questions,
                    quiz=quiz,
                    tokensUsedTotal=TokensUsed(input=v_in + q_in, output=v_out + q_out),
                    latencyMsTotal=v_lat + q_lat
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