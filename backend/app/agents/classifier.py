"""Classification Agent — multi-label clause risk classification using LoRA-adapted SmolLM3."""

import os
import json
import re
import httpx
from openai import OpenAI


SYSTEM_PROMPT = "You are a legal risk classification expert."

USER_PROMPT_TEMPLATE = """Identify all applicable risk categories and sub-categories present in this clause. Output as a JSON list. Each risk should have a 'risk' field and a 'sub_category' field.

ORIGINAL CLAUSE:
\"\"\"
{clause_text}
\"\"\""""


class ClassificationAgent:
    """Classification Agent — multi-label clause risk classification using LoRA-adapted SmolLM3.

    Pipeline step: 3 (reference Section 5 of PLAN.md)
    """

    def __init__(self, model_name: str = "LawUP-Classifier", lm_studio_url: str = None):
        if lm_studio_url is None:
            lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://172.25.241.3:1234/v1")
        self.client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
        self.model_name = model_name
        self.url = lm_studio_url

    def run(self, clause_text: str) -> list[dict]:
        """Classifies the given clause into zero or more risk categories.
        
        Args:
            clause_text: The raw legal clause to classify.
            
        Returns:
            A list of dicts, e.g. [{"risk": "Termination Risk", "sub_category": "..."}]
        """
        prompt = USER_PROMPT_TEMPLATE.format(clause_text=clause_text.strip())

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # Zero temperature for deterministic extraction
            )
            raw_text = response.choices[0].message.content.strip()
            
            # Use regex to extract JSON list in case the model adds extra chat formatting
            json_match = re.search(r'(\[.*\])', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = raw_text
                
            return json.loads(json_str)

        except httpx.ConnectError:
            print(f"ERROR: Could not connect to LM Studio server at {self.url}.")
            raise
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from classifier model: {raw_text}")
            # If parsing fails, return empty risks to fail safely
            return []
        except Exception as e:
            print(f"ERROR calling classification agent: {e}")
            return []


if __name__ == "__main__":
    # Test standalone execution
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://172.25.241.3:1234/v1", help="LM Studio API URL")
    args = parser.parse_args()

    agent = ClassificationAgent(lm_studio_url=args.url)
    
    test_clause = (
        "The Agreement shall be governed by and construed under the laws of the State of Florida "
        "in the United States of America, and venue for any such legal action shall be in the "
        "Circuit Court or County Court in Orlando, FL."
    )
    
    print(f"Testing Classification Agent on LM Studio ({args.url})...")
    print("\nClause:", test_clause)
    
    risks = agent.run(test_clause)
    print("\nIdentified Risks:")
    print(json.dumps(risks, indent=2))
