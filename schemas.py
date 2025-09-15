from typing import Annotated, List, Literal, Optional, Dict
from pydantic import BaseModel, StringConstraints

StrMin1 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BodyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=50)]

NewsStyle = Literal["CONCISE", "FRIENDLY", "NEUTRAL"]

class RewriteRequest(BaseModel):
    articleId: StrMin1
    title: StrMin1
    body: BodyStr

class TokensUsed(BaseModel):
    input: int = 0
    output: int = 0

class Quiz(BaseModel):
    question: StrMin1
    answer: Literal["YES", "NO"]  # 대문자 YES/NO만 허용

# ------- EPI 평가 모델 ------
class EpiResult(BaseModel):
    epiOriginal: int
    epiSummary: int
    reductionPct: float
    stimulationReduced: str
    componentsOriginal: Dict[str, float]
    componentsSummary: Dict[str, float]
    reason: str

# ------- 뉴스 요약 응답 -----
class RewriteResponse(BaseModel):
    articleId: str
    newTitle: str
    summary: str
    questions: List[str]
    quiz: Quiz
    tokensUsed: TokensUsed
    model: str
    latencyMs: int
    epi: EpiResult

# ------- 배치 전용(느슨하게: body 길이 제한 없음) -------
class RewriteBatchItemIn(BaseModel):
    articleId: StrMin1
    title: StrMin1
    body: str  # 길이 제한 제거 (엔드포인트 내부에서 검사)

class RewriteBatchRequest(BaseModel):
    items: List[RewriteBatchItemIn]

class RewriteBatchItemResult(BaseModel):
    articleId: str
    ok: bool
    data: Optional[RewriteResponse] = None
    error: Optional[str] = None

class RewriteBatchResponse(BaseModel):
    results: List[RewriteBatchItemResult]

# ------- 챗 봇 -------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatArticleRequest(BaseModel):
    articleId: str
    userId: str
    summary: str
    history: List[ChatMessage]
    userMessage: str

class ChatArticleResponse(BaseModel):
    articleId: str
    userId: str
    userMessage: str
    answer: str
    model: str
    latencyMs: int

class RewriteVariant(BaseModel):
    newsStyle: NewsStyle
    articleId: str
    newTitle: str
    summary: str
    questions: List[str]
    quiz: Quiz
    tokensUsed: TokensUsed
    model: str
    latencyMs: int
    epi: EpiResult

class RewriteMultiResponse(BaseModel):
    articleId: str
    variants: List[RewriteVariant]          # 길이 3 (스타일별)
    tokensUsedTotal: TokensUsed             # variants 합계
    latencyMsTotal: int                     # variants 합계(또는 max)

# 배치용 결과도 멀티 버전을 추가
class RewriteBatchItemMultiResult(BaseModel):
    articleId: str
    ok: bool
    data: Optional[RewriteMultiResponse] = None
    error: Optional[str] = None

class RewriteBatchMultiResponse(BaseModel):
    results: List[RewriteBatchItemMultiResult]