#This file contains the Pydantic models used for request and response validation in the API.


from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    Data received from the user.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question asked by the user."
    )


class SourceDocument(BaseModel):
    """
    Information about a retrieved document chunk.
    """

    source: str
    chunk: str


class QuestionResponse(BaseModel):
    """
    Final API response.
    """

    question: str
    answer: str
    retrieved_chunks: list[SourceDocument]