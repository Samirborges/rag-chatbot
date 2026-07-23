from src.evaluation.ragas_eval import make_eval_record


def test_make_eval_record_contains_required_fields():
    record = make_eval_record(
        question="O que é RAG?",
        answer="RAG combina recuperação e geração.",
        contexts=["contexto 1", "contexto 2"],
        ground_truth="RAG combina recuperação e geração.",
    )

    assert record["question"] == "O que é RAG?"
    assert record["answer"] == "RAG combina recuperação e geração."
    assert record["contexts"] == ["contexto 1", "contexto 2"]
    assert record["ground_truth"] == "RAG combina recuperação e geração."
