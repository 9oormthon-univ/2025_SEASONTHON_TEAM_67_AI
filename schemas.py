from typing import Annotated, List
from pydantic import BaseModel, StringConstraints

StrMin1 = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BodyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=50)]

class RewriteRequest(BaseModel):
    articleId: StrMin1
    title: StrMin1
    body: BodyStr

class TokensUsed(BaseModel):
    input: int = 0
    output: int = 0

class RewriteResponse(BaseModel):
    articleId: str
    newTitle: str
    summary: str
    questions: List[str]
    tokensUsed: TokensUsed
    model: str
    latencyMs: int
