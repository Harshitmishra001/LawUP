import os
os.environ["HF_TOKEN"] = ""
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import json
import pytest
from pathlib import Path
from backend.app.retrieval import GroundingRetriever

@pytest.fixture(scope="module")
def retriever():
    return GroundingRetriever()

@pytest.fixture(scope="module")
def eval_set():
    path = Path("backend/tests/retrieval_eval_set.json")
    if not path.exists():
        pytest.skip(f"Eval set {path} not found. Skipping retrieval test.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_retrieval_precision(retriever, eval_set):
    correct_hits = 0
    total = len(eval_set)
    
    for item in eval_set:
        results = retriever.retrieve_grounding(
            risk_category=item["risk_category"],
            clause_text=item["query_clause"],
            top_k=1
        )
        
        # We expect exactly 1 result since top_k=1
        assert len(results) > 0, f"No results found for {item['query_clause']}"
        
        retrieved_sub_cat = results[0]["sub_category"]
        expected_sub_cat = item["expected_sub_category"]
        
        if retrieved_sub_cat == expected_sub_cat:
            correct_hits += 1
        else:
            print(f"Failed retrieval for: {item['query_clause']}")
            print(f"Expected: {expected_sub_cat}, Got: {retrieved_sub_cat}")
            
    accuracy = correct_hits / total
    print(f"\nRetrieval Accuracy: {accuracy * 100:.2f}% ({correct_hits}/{total})")
    
    # Assert at least 90% accuracy on this curated eval set
    assert accuracy >= 0.90
