import json
import sys
from pathlib import Path

def validate_corpus():
    corpus_path = Path("grounding_corpus/grounding_corpus.json")
    
    if not corpus_path.exists():
        print(f"NOTE: '{corpus_path}' does not exist yet. Skipping schema validation gracefully as M4 work is in progress.")
        sys.exit(0)
        
    try:
        with corpus_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to parse '{corpus_path}' as JSON. {e}")
        sys.exit(1)
        
    if not isinstance(data, list):
        print(f"ERROR: Expected '{corpus_path}' to contain a JSON array (list).")
        sys.exit(1)
        
    required_fields = {"risk_category", "sub_category", "reference_text", "authority_type", "scope"}
    errors_found = False
    
    for i, entry in enumerate(data):
        missing_fields = required_fields - set(entry.keys())
        if missing_fields:
            print(f"ERROR [Entry {i}]: Missing required fields: {missing_fields}")
            errors_found = True
            
        authority_type = entry.get("authority_type")
        if authority_type != "industry_norm":
            print(f"ERROR [Entry {i}]: 'authority_type' must be exactly 'industry_norm' for Phase 1. Found: '{authority_type}'")
            errors_found = True
            
        reference_text = entry.get("reference_text")
        if reference_text is not None and not str(reference_text).strip():
            print(f"ERROR [Entry {i}]: 'reference_text' cannot be empty.")
            errors_found = True
            
    if errors_found:
        print("\nSchema validation failed due to the errors above.")
        sys.exit(1)
        
    print(f"SUCCESS: '{corpus_path}' schema is valid.")

if __name__ == "__main__":
    validate_corpus()
