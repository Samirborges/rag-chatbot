import logging

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSerializable

from src.retrieval.retriever import get_retriever
from src.generation.prompt_builder import build_prompt, format_docs
from src.generation.llm_client import get_llm

logger = logging.getLogger(__name__)


def build_rag_chain() -> RunnableSerializable:
    """
    Constrói a chain RAG completa (retriever -> prompt -> llm -> parser).
    Deve ser chamada UMA VEZ na inicialização da aplicação, não a cada pergunta.
    """
    retriever = get_retriever()
    prompt = build_prompt()
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain construída com sucesso.")
    return chain


def answer_question(chain: RunnableSerializable, query: str) -> str:
    """
    Executa uma pergunta contra uma chain RAG já construída.
    A chain deve ser construída uma única vez (via build_rag_chain)
    e reutilizada entre chamadas.
    """
    logger.info("Pergunta recebida: %s", query)
    response = chain.invoke(query)
    return response

