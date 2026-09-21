import logging
from typing import Any

from openai import OpenAI

from app.config import Settings


logger = logging.getLogger(__name__)


class LLMService:
    """
    Handles communication with the OpenAI Responses API.
    """

    def __init__(self, settings: Settings):

        self.settings = settings

        self.client: OpenAI | None = None

        if self.settings.OPENAI_API_KEY:

            self.client = OpenAI(
                api_key=self.settings.OPENAI_API_KEY
            )

            logger.info(
                "OpenAI client initialized."
            )

        else:

            logger.warning(
                "OPENAI_API_KEY not found. "
                "Application will use retrieval-only mode."
            )

    # -----------------------------------------------------
    # GENERATE ANSWER
    # -----------------------------------------------------

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:

        if not retrieved_chunks:

            return (
                "I could not find relevant information "
                "in the company documents."
            )

        # -------------------------------------------------
        # BUILD CONTEXT
        # -------------------------------------------------

        context_parts = []

        for item in retrieved_chunks:

            metadata = item.get(
                "metadata",
                {},
            )

            filename = metadata.get(
                "filename",
                "Unknown source",
            )

            text = item.get(
                "text",
                "",
            )

            context_parts.append(
                f"Source: {filename}\n{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        # -------------------------------------------------
        # RETRIEVAL-ONLY MODE
        # -------------------------------------------------

        if self.client is None:

            return (
                "No OPENAI_API_KEY was configured, "
                "so I am returning the most relevant "
                "retrieved information:\n\n"
                f"{context}"
            )

        # -------------------------------------------------
        # SYSTEM INSTRUCTIONS
        # -------------------------------------------------

        instructions = """
You are a company policy assistant.

Your job is to answer the user's question using ONLY
the company information provided in the context.

Rules:

1. Do not invent company policies.
2. Do not use information that is not supported by the context.
3. If the answer is not available, clearly say:
   "I could not find that information in the company documents."
4. Give a direct and easy-to-understand answer.
5. When useful, mention the source document.
""".strip()

        user_input = f"""
COMPANY DOCUMENT CONTEXT

{context}

USER QUESTION

{question}
""".strip()

        # -------------------------------------------------
        # CALL OPENAI
        # -------------------------------------------------

        try:

            response = self.client.responses.create(
                model=self.settings.OPENAI_MODEL,
                instructions=instructions,
                input=user_input,
            )

            return response.output_text.strip()

        except Exception as exc:

            logger.exception(
                "OpenAI request failed."
            )

            raise RuntimeError(
                "The AI service could not generate an answer."
            ) from exc