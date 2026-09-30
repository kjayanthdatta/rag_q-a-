"""Pydantic request/response models."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="User question about the document")

    def cleaned_question(self) -> str:
        return self.question.strip()


class SourceChunk(BaseModel):
    page: int
    content: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks: int
