from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset

from src.generation.llm_client import get_llm
from src.ingestion.embedder import get_vectorstore
from src.pipeline import answer_question, build_rag_chain
from src.retrieval.retriever import retrieve_relevant_chunks

logger = logging.getLogger(__name__)


def make_eval_record(
    question: str,
    answer: str,
    contexts: Sequence[str],
    ground_truth: str | None = None,
) -> dict[str, Any]:
    """Normaliza um registro para ser consumido pelo RAGAS."""
    record: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "contexts": list(contexts),
    }
    if ground_truth is not None:
        record["ground_truth"] = ground_truth
    return record


def build_evaluation_records(
    questions: Sequence[str],
    ground_truths: Sequence[str] | None = None,
    session_id_prefix: str = "ragas-eval",
) -> list[dict[str, Any]]:
    """Executa o pipeline RAG para cada pergunta e devolve os registros de avaliação."""
    chain = build_rag_chain()
    records: list[dict[str, Any]] = []

    for index, question in enumerate(questions):
        answer = answer_question(chain, question, session_id=f"{session_id_prefix}-{index}")
        documents = retrieve_relevant_chunks(question, k=4)
        contexts = [doc.page_content for doc in documents]
        ground_truth = None if ground_truths is None else ground_truths[index]
        records.append(make_eval_record(question, answer, contexts, ground_truth=ground_truth))

    return records


def run_ragas_evaluation(
    questions: Sequence[str],
    ground_truths: Sequence[str] | None = None,
    session_id_prefix: str = "ragas-eval",
) -> dict[str, Any]:
    """
    Executa uma avaliação completa do pipeline RAG usando métricas RAGAS.

    Requer `ragas` e `datasets` instalados no ambiente Python do projeto.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS não está disponível no ambiente. Instale `ragas` e `datasets` primeiro."
        ) from exc

    records = build_evaluation_records(
        questions=questions,
        ground_truths=ground_truths,
        session_id_prefix=session_id_prefix,
    )

    dataset = Dataset.from_list(records)
    llm = get_llm()
    vector_store = get_vectorstore()
    embedding_model = getattr(vector_store, "embeddings", None)
    if embedding_model is None:
        embedding_model = getattr(vector_store, "embedding", None)

    results = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embedding_model,
    )

    return {
        "records": records,
        "results": results,
    }


def _serialize_ragas_result(result_obj: Any) -> dict[str, Any]:
    """Converte o objeto `EvaluationResult` do RAGAS em um formato JSON-safe."""
    if hasattr(result_obj, "to_pandas"):
        try:
            dataframe = result_obj.to_pandas()
            rows = dataframe.to_dict(orient="records")
            return {
                "rows": rows,
                "summary": getattr(result_obj, "_repr_dict", {}),
            }
        except Exception:
            pass

    if hasattr(result_obj, "scores"):
        try:
            return {
                "rows": list(result_obj.scores),
                "summary": getattr(result_obj, "_repr_dict", {}),
            }
        except Exception:
            pass

    if isinstance(result_obj, dict):
        return result_obj

    return {"value": str(result_obj)}


def save_eval_result(result: dict[str, Any], output_path: str | Path) -> Path:
    """Salva o resultado em JSON para inspeção posterior."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    serialized = {
        "records": result["records"],
        "results": _serialize_ragas_result(result["results"]),
    }

    output_file.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Resultado salvo em %s", output_file)
    return output_file
