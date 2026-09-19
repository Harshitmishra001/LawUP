import json
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import torch

class GroundingRetriever:
    def __init__(self, corpus_path: str = "grounding_corpus/grounding_corpus.json", model_name: str = "all-MiniLM-L6-v2"):
        self.corpus_path = Path(corpus_path)
        self.model = SentenceTransformer(model_name)
        self.corpus = self._load_corpus()
        
        # Pre-compute embeddings for the reference texts
        self.reference_texts = [item["reference_text"] for item in self.corpus]
        if self.reference_texts:
            self.corpus_embeddings = self.model.encode(self.reference_texts, convert_to_tensor=True)
        else:
            self.corpus_embeddings = None

    def _load_corpus(self) -> List[Dict[str, Any]]:
        # Handle path resolution assuming this might be run from project root or backend dir
        paths_to_try = [
            self.corpus_path,
            Path("..") / self.corpus_path,
            Path("../..") / self.corpus_path,
            Path("../../..") / self.corpus_path,
        ]
        
        for p in paths_to_try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
                    
        raise FileNotFoundError(f"Could not find {self.corpus_path} in standard locations.")

    def retrieve_grounding(self, risk_category: str, clause_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant industry norm/statute for a given risk category and clause.
        Filters by risk_category first to ensure we don't retrieve IP norms for a Termination risk.
        """
        if self.corpus_embeddings is None or len(self.corpus) == 0:
            return []

        # Filter indices by risk category (so we only compare against relevant norms)
        # We do soft matching on risk_category just in case
        valid_indices = [
            i for i, item in enumerate(self.corpus) 
            if item["risk_category"] == risk_category
        ]
        
        if not valid_indices:
            return []
            
        # Extract the embeddings for just the valid indices
        valid_embeddings = self.corpus_embeddings[valid_indices]
        
        # Encode the query (the actual contract clause)
        query_embedding = self.model.encode(clause_text, convert_to_tensor=True)
        
        # Compute cosine similarities
        cos_scores = util.cos_sim(query_embedding, valid_embeddings)[0]
        
        # Get top_k results
        k = min(top_k, len(cos_scores))
        top_results = torch.topk(cos_scores, k=k)
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            original_idx = valid_indices[idx.item()]
            result_item = self.corpus[original_idx].copy()
            result_item["similarity_score"] = float(score.item())
            results.append(result_item)
            
        return results

if __name__ == "__main__":
    retriever = GroundingRetriever()
    print("Testing Retrieval:")
    res = retriever.retrieve_grounding(
        risk_category="Non-Compete / Exclusivity Overreach",
        clause_text="Contractor agrees that for a period of 2 years after termination, they will not work for any competitor."
    )
    print(json.dumps(res, indent=2))
