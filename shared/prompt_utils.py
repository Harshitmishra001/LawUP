def format_prompt(original_clause: str) -> str:
    """
    EXACT formatting string used during M1/M2 training via data/scripts/format_for_qlora.py.
    This must match precisely to avoid silent quality degradation.
    """
    prompt = (
        "You are an expert legal assistant. Your task is to rewrite the following complex legal clause "
        "into plain, easy-to-understand English. Ensure that you preserve all original obligations, "
        "conditions, and permissions without adding or removing any meaning.\n\n"
        f"Original Clause:\n{original_clause}\n\n"
        "Plain English Rewrite:\n"
    )
    return prompt
