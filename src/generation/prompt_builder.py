from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


SYSTEM_TEMPLATE = """Você é um assistente que responde perguntas EXCLUSIVAMENTE
com base no contexto abaixo, extraído do material do curso.

Se a resposta não estiver no contexto, diga que não encontrou
essa informação no material do curso — não invente.

Contexto:
{context}
"""

CONTEXTUALIZE_SYSTEM_TEMPLATE = """Dado o histórico da conversa e a pergunta mais recente
do usuário, reformule a pergunta para que ela seja compreendida de forma
independente, sem necessidade do histórico. Não responda a pergunta,
apenas reformule-a se necessário."""


def build_prompt() -> ChatPromptTemplate:
    """Prompt final: recebe contexto recuperado, histórico e a pergunta."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])


def build_contextualize_prompt() -> ChatPromptTemplate:
    """
    Prompt usado só para reformular a pergunta com base no histórico,
    antes de mandar pro retriever. Chaves 'chat_history' e 'input' são
    exigidas pela implementação do create_history_aware_retriever.
    """
    return ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def format_docs(documents) -> str:
    docs_formatados = ""
    for doc in documents:
        docs_formatados += f"{doc.page_content}\n\n{doc.metadata}\n\n---\n\n"
    return docs_formatados

