import logging
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory

from src.retrieval.retriever import get_retriever
from src.generation.prompt_builder import build_prompt, build_contextualize_prompt, format_docs
from src.generation.llm_client import get_llm

logger = logging.getLogger(__name__)

store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_rag_chain() -> RunnableSerializable:
    """
    Constrói a chain RAG completa, com memória de conversação.
    Deve ser chamada UMA VEZ na inicialização da aplicação.
    """
    retriever = get_retriever()
    llm = get_llm()

    contextualize_prompt = build_contextualize_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    prompt = build_prompt()

    # RunnablePassthrough.assign mantém as chaves já existentes no input
    # (question, history) e ADICIONA uma nova chave "context" ao dicionário.
    chain_base = (
        RunnablePassthrough.assign(
            context=(
                {
                    "input": itemgetter("question"),
                    "chat_history": itemgetter("history"),
                }
                | history_aware_retriever
                | format_docs
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_com_memoria = RunnableWithMessageHistory(
        chain_base,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    logger.info("RAG chain (com memória) construída com sucesso.")
    return chain_com_memoria


def answer_question(chain: RunnableSerializable, query: str, session_id: str) -> str:
    """
    Executa uma pergunta contra a chain RAG com memória.
    session_id identifica a conversa (permite múltiplas sessões simultâneas).
    """
    logger.info("[session=%s] Pergunta recebida: %s", session_id, query)
    response = chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": session_id}},
    )
    return response

