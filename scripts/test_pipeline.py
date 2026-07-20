import logging

from src.pipeline import build_rag_chain, answer_question

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    chain = build_rag_chain()

    perguntas_teste = [
        "O que é RAG?",
        "Qual a diferença entre busca vetorial e busca por palavra-chave?",
    ]

    for pergunta in perguntas_teste:
        print("=" * 60)
        print(f"PERGUNTA: {pergunta}")
        resposta = answer_question(chain, pergunta)
        print(f"RESPOSTA: {resposta}")