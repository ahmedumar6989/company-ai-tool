import logging

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from app.config import settings
from app.schemas.questions import (
    QuestionRequest,
    QuestionResponse,
    SourceDocument,
)


logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@router.get("/")
def home():
    return {
        "message": "Company AI Assistant is running.",
        "endpoint": "POST /ask",
        "documentation": "/docs",
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }


# ---------------------------------------------------------
# ASK QUESTION
# ---------------------------------------------------------

@router.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask_question(
    request: Request,
    payload: QuestionRequest,
):
    question = payload.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retrieval_service = (
        request.app.state.retrieval_service
    )

    llm_service = (
        request.app.state.llm_service
    )

    try:

        # -------------------------------------------------
        # RETRIEVE RELEVANT DOCUMENTS
        # -------------------------------------------------

        retrieved_chunks = (
            retrieval_service.retrieve(
                question=question,
                top_k=settings.TOP_K,
            )
        )

        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        answer = llm_service.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # -------------------------------------------------
        # FORMAT SOURCES
        # -------------------------------------------------

        sources = []

        for item in retrieved_chunks:

            metadata = item.get(
                "metadata",
                {},
            )

            sources.append(
                SourceDocument(
                    source=metadata.get(
                        "filename",
                        "Unknown source",
                    ),
                    chunk=item.get(
                        "text",
                        "",
                    ),
                )
            )

        # -------------------------------------------------
        # RETURN RESPONSE
        # -------------------------------------------------

        return QuestionResponse(
            question=question,
            answer=answer,
            retrieved_chunks=sources,
        )

    except RuntimeError as exc:

        logger.error(
            "Application error: %s",
            exc,
        )

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected error while processing question."
        )

        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from exc