import logging
import uuid

from src.pipeline import build_rag_chain, answer_question


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    chain = build_rag_chain()
    session_id = str(uuid.uuid4())

    print("Bem-vindo(a)!")

    while True:
        question = input("Você: ")
        if question == "sair" or question == "exit":
            break
            
        try:
            response = answer_question(chain, question, session_id)
            print(f"Bot: {response}")
        except Exception as e:
            logger.error("[ERRO]: %s", e)
            print("Ocorreu um erro inesperado. Tente novamente mais tarde")


