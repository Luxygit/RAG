"""validating via pydantic models"""


from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """this will hold a single retrieved text snippet"""
    file_path: str = Field(..., min_length=2)
    first_character_index: int = Field(..., ge=0)
    last_character_index: int = Field(..., ge=0)


class UnansweredQuestion(BaseModel):
    """this is a before-response question"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(..., min_length=2)


class AnsweredQuestion(UnansweredQuestion):
    """this is the question plus the answer and sources"""
    sources: list[MinimalSource]
    answer: str = Field(..., min_length=2)


class RagDataset(BaseModel):
    """represents a set of RAG questions"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """search results"""
    question_id: str | None
    question: str | None
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """represents an answer"""
    answer: str


class StudentSearchResults(BaseModel):
    """represents search results"""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswers(BaseModel):
    """this will be a search result with answers"""
    search_results: list[MinimalAnswer]
    k: int
