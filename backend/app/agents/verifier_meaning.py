import os
import json
import httpx
import re
from openai import OpenAI

SYSTEM_PROMPT = """You are a meaning-preservation auditor for legal contract simplification. 
You will be given two texts:

- ORIGINAL: a clause from a real legal contract, written in formal legal language.
- REWRITE: a plain-English rewrite of that clause, produced by an AI model.

Your job is NOT to judge writing quality, tone, or readability. Your job is 
to determine whether the REWRITE preserves the exact legal meaning of the 
ORIGINAL — no more, no less.

You must check meaning preservation in BOTH directions:

DIRECTION A — Forward entailment (checks for hallucination/contradiction):
  Assume ORIGINAL is true. Does everything asserted in REWRITE necessarily 
  follow? Flag anything in REWRITE that is NOT supported by ORIGINAL — 
  including invented conditions, inverted obligations (e.g. "may" 
  rewritten as "must"), added parties, added dollar amounts, or merged 
  provisions that change who a duty applies to.

DIRECTION B — Backward entailment (checks for omission):
  Assume REWRITE is true. Does everything asserted in ORIGINAL necessarily 
  follow? Flag anything in ORIGINAL that is NOT captured in REWRITE — 
  including dropped conditions, dropped sub-clauses, dropped qualifiers 
  ("only," "except," "provided that"), narrowed scope, or collapsed 
  distinctions between two separate provisions (e.g. two parties with 
  different rights being merged into one shared right).

A rewrite only PASSES if BOTH directions hold. If either direction fails, 
the rewrite FAILS, regardless of how well-written it is.

Classify every failure into exactly one category:
- HALLUCINATION: REWRITE asserts something ORIGINAL does not support.
- CONTRADICTION: REWRITE inverts or reverses an obligation, right, or 
  condition from ORIGINAL.
- OMISSION: REWRITE drops a standalone condition, qualifier, or obligation from ORIGINAL without merging anything.
- STRUCTURAL_COLLAPSE: REWRITE merges two or more distinct provisions (e.g. different rights held by different parties, or separate conditions) into one, losing the specific boundary between them.

A rewrite may have multiple failures. List all of them.

Do not consider stylistic simplification, reordering of sentences, or 
converting passive to active voice as failures, as long as the legal 
substance is unchanged.

You must respond with ONLY a single valid JSON object. No preamble, no 
markdown code fences, no explanation outside the JSON fields below.

Output schema:
{
  "original_entails_rewrite": boolean,
  "rewrite_entails_original": boolean,
  "verdict": "PASS" | "FAIL",
  "failures": [
    {
      "category": "HALLUCINATION" | "CONTRADICTION" | "OMISSION" | "STRUCTURAL_COLLAPSE",
      "description": "One or two sentences pinpointing exactly what changed and where in the text.",
      "original_span": "The specific phrase or sub-clause in ORIGINAL involved (quote sparingly, just enough to locate it)",
      "rewrite_span": "The specific phrase in REWRITE involved, or null if the issue is a pure omission with no corresponding text"
    }
  ],
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}

If verdict is "PASS", "failures" must be an empty array.
Set confidence to LOW if the clause contains cross-references (e.g. 
"as provided in Article 12") that you cannot fully resolve without 
additional context, or if the clause is long enough that subtle 
omissions are hard to rule out with certainty."""

USER_MESSAGE_TEMPLATE = """ORIGINAL:
{original_clause}

REWRITE:
{rewrite_text}

Evaluate meaning preservation per your instructions. Respond with the JSON object only."""

# Setup LM Studio client
lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")

def verify_meaning(original: str, rewrite: str, model: str = "LawUP") -> dict:
    prompt = USER_MESSAGE_TEMPLATE.format(original_clause=original, rewrite_text=rewrite)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        raw_text = response.choices[0].message.content.strip()
        
        # Robust JSON extraction using regex
        json_match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_text
            
        return json.loads(json_str)
        
    except httpx.ConnectError:
        print(f"ERROR: Could not connect to LM Studio server at {lm_studio_url}.")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON from model: {raw_text}")
        return {"error": "JSON parse failed", "raw": raw_text}
    except Exception as e:
        print(f"ERROR calling verifier: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test standalone on Example 3
    test_original = """Notwithstanding the foregoing, (a) either Party may, without the other Party's consent, assign this Agreement and its rights and obligations hereunder in whole or in part to an Affiliate; and (b) Dova may assign this Agreement to a successor in interest in connection with the sale or other transfer of all or substantially all of Dova's assets or rights relating to the Product; provided that such assignee shall remain subject to all of the terms and conditions hereof in all respects and shall assume all obligations of Dova hereunder whether accruing before or after such assignment."""
    
    test_rewrite = """Either Party may, without the other's consent, transfer this Agreement and its duties under it to an Affiliate. However:

(a) if a successor takes over Dova's assets or rights related to the Product in connection with a sale or other transfer of all or substantially all of those assets or rights, then Dova may assign this Agreement to that successor.

In either case, the new party must follow all terms and conditions of this Agreement. The new party also assumes all duties owed by Dova under this Agreement, whether they were created before or after the assignment."""
    
    print("Testing Example 3 Standalone...")
    result = verify_meaning(test_original, test_rewrite)
    print(json.dumps(result, indent=2))
