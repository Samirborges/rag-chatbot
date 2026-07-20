from langchain_core.prompts import ChatPromptTemplate


TEMPLATE = """
Você é um assistente que responde perguntas EXCLUSIVAMENTE
com base no contexto abaixo, extraído do material do curso.

Se a resposta não estiver no contexto, diga que não encontrou
essa informação no material do curso — não invente.

Contexto:
{context}

Pergunta:
{question}

"""

def build_prompt():
    return ChatPromptTemplate.from_template(TEMPLATE)


def format_docs(documents):
    
    documents_formated = ""
    
    for doc in documents:
        documents_formated += f"{doc.page_content} \n\n {doc.metadata}  \n\n---\n\n"
    
    return documents_formated