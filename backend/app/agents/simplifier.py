"""Simplifier Agent — generates plain-language rewrites via LoRA-adapted SmolLM3."""

import os
import httpx
from openai import AsyncOpenAI, OpenAI

from shared.prompt_utils import format_prompt


class SimplifierAgent:
    """Simplifier Agent — generates plain-language rewrites via LoRA-adapted SmolLM3.

    Pipeline step: 2 (reference Section 5 of PLAN.md)

    Uses the completion API (not chat) with greedy decoding and an explicit
    <|endoftext|> stop token to prevent run-on generation — matching exactly how
    the model was fine-tuned.
    """

    def __init__(
        self,
        model_name: str = "lawup-3b-q8",
        lm_studio_url: str = None,
        max_prompt_tokens: int = 1024,
        max_output_tokens: int = 512,
    ):
        if lm_studio_url is None:
            lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        self.url = lm_studio_url
        self.model_name = model_name
        self.max_prompt_tokens = max_prompt_tokens
        self.max_output_tokens = max_output_tokens

        # Synchronous client for blocking calls
        self._sync_client = OpenAI(base_url=lm_studio_url, api_key="lm-studio")
        # Async client for FastAPI endpoints
        self._async_client = AsyncOpenAI(base_url=lm_studio_url, api_key="lm-studio")

    def run(self, clause_text: str) -> str:
        """Synchronously simplify a clause. Returns the plain-English rewrite string.

        Args:
            clause_text: The raw legal clause text.

        Returns:
            Plain-English rewrite as a string, or raises on connection failure.
        """
        if not clause_text or not clause_text.strip():
            raise ValueError("clause_text cannot be empty.")

        prompt = format_prompt(clause_text.strip())

        try:
            response = self._sync_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_output_tokens,
                temperature=0.0,  # Greedy decoding — deterministic output
                stop=["<|endoftext|>"],  # Match fine-tuning stop token
            )
            return response.choices[0].text.strip()
        except httpx.ConnectError:
            print(f"ERROR: Could not connect to LM Studio at {self.url}.")
            raise

    async def run_async(self, clause_text: str) -> str:
        """Asynchronously simplify a clause. Used by FastAPI endpoints.

        Args:
            clause_text: The raw legal clause text.

        Returns:
            Plain-English rewrite as a string, or raises on connection failure.
        """
        if not clause_text or not clause_text.strip():
            raise ValueError("clause_text cannot be empty.")

        prompt = format_prompt(clause_text.strip())

        try:
            response = await self._async_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_output_tokens,
                temperature=0.0,
                stop=["<|endoftext|>"],
            )
            return response.choices[0].text.strip()
        except httpx.ConnectError:
            print(f"ERROR: Could not connect to LM Studio at {self.url}.")
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:1234/v1", help="LM Studio API URL")
    args = parser.parse_args()

    agent = SimplifierAgent(lm_studio_url=args.url)

    test_clause = (
        "The Receiving Party shall indemnify and hold harmless the Disclosing Party "
        "against any consequential damages arising from a breach."
    )

    print(f"Testing SimplifierAgent on LM Studio ({args.url})...")
    print("\nClause:", test_clause)
    rewrite = agent.run(test_clause)
    print("\nRewrite:", rewrite)
