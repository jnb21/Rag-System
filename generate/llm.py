"""Generation: send the augmented prompt to Claude via LangChain."""

import os

from langchain_core.messages import HumanMessage

MODEL = "claude-haiku-4-5-20251001"


def generate(prompt, fallback_text):
    """Call Claude with the assembled prompt, or fall back if no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return f"(extractive fallback, no ANTHROPIC_API_KEY set) Most relevant passage: {fallback_text}"

    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=MODEL, max_tokens=300)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
