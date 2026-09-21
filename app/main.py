import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import (
    ensure_directories,
    settings,
)
from app.services.ingestion import (
    DocumentIngestionService,
)
from app.services.llm import LLMService
from app.services.retrieval import (
    RetrievalService,
)


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# APPLICATION STARTUP
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Starting %s...",
        settings.APP_NAME,
    )

    # -----------------------------------------------------
    # CREATE REQUIRED DIRECTORIES
    # -----------------------------------------------------

    ensure_directories()

    # -----------------------------------------------------
    # RETRIEVAL SERVICE
    # -----------------------------------------------------

    retrieval_service = RetrievalService(
        settings
    )

    # -----------------------------------------------------
    # DOCUMENT INGESTION
    # -----------------------------------------------------

    ingestion_service = (
        DocumentIngestionService(
            settings=settings,
            retrieval_service=retrieval_service,
        )
    )

    ingestion_service.build_index()

    # -----------------------------------------------------
    # LLM SERVICE
    # -----------------------------------------------------

    llm_service = LLMService(
        settings
    )

    # -----------------------------------------------------
    # STORE SERVICES IN FASTAPI STATE
    # -----------------------------------------------------

    app.state.retrieval_service = (
        retrieval_service
    )

    app.state.llm_service = (
        llm_service
    )

    logger.info(
        "Application startup completed."
    )

    yield

    # -----------------------------------------------------
    # SHUTDOWN
    # -----------------------------------------------------

    logger.info(
        "%s stopped.",
        settings.APP_NAME,
    )


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A Retrieval-Augmented Generation API "
        "for answering questions from company documents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# REGISTER ROUTES
# ---------------------------------------------------------

app.include_router(router)