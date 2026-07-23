import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.ragas_eval import run_ragas_evaluation, save_eval_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


if __name__ == "__main__":
    questions = [
        "O que é RAG?",
        "Qual a diferença entre busca vetorial e busca por palavra-chave?",
        "Como o retriever escolhe os chunks mais relevantes?",
    ]

    ground_truths = [
        "RAG é uma arquitetura que combina recuperação de documentos com geração de resposta usando um modelo de linguagem.",
        "Busca vetorial usa similaridade semântica por embeddings; busca por palavra-chave faz correspondência textual literal.",
        "O retriever usa similaridade entre embeddings da pergunta e dos chunks para selecionar os trechos mais relevantes.",
    ]

    result = run_ragas_evaluation(
        questions=questions,
        ground_truths=ground_truths,
        session_id_prefix="eval-ragas",
    )

    output_path = Path("data") / "evaluation" / "ragas_results.json"
    save_eval_result(result, output_path)
