from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from datetime import datetime
import math


# Custom Tools 

def get_current_datetime() -> dict:
    """Returns the current date and time."""
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day": now.strftime("%A"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S")
    }

def calculator(expression: str) -> dict:
    """
    Evaluates a mathematical expression safely.
    Args:
        expression: A math expression like '2+2', '10*5', 'sqrt(16)'
    Returns:
        Result of the calculation
    """
    try:
        result = eval(expression, {"math": math})

        return {
            "expression": expression,
            "result": result,
            "status": "success"
        }

    except Exception as e:

        return {
            "expression": expression,
            "error": str(e),
            "status": "failed"
        }


def summarize_text(text: str, max_words: int = 50) -> dict:
    """
    Summarizes a given text to a specified word limit.
    Args:
        text: The text to summarize
        max_words: Maximum words in summary (default 50)
    Returns:
        Summarized text
    """
    words = text.split()
    if len(words) <= max_words:
        return {"summary": text, "original_words": len(words), "summary_words": len(words)}
    summary = " ".join(words[:max_words]) + "..."
    return {
        "summary": summary,
        "original_words": len(words),
        "summary_words": max_words
    }


#  Root Agent 
root_agent = Agent(
    name="tools_agent",
    description="A smart AI assistant that can search the web, do calculations, tell date/time, and summarize text.",
    instruction="""
        You are a helpful AI assistant.

        You can answer general knowledge questions using your built-in knowledge.

        You also have access to these tools:
        1. get_current_datetime - use when user asks date or time
        2. calculator - use for calculations
        3. summarize_text - use for summarizing long text

        Rules:
        - Answer normal questions directly.
        - Only use tools when they are actually needed.
        - Do not say you only know about your tools.
        - Be friendly and concise.
""",
    model="gemini-2.5-flash",
    tools=[
        FunctionTool(get_current_datetime),
        FunctionTool(calculator),
        FunctionTool(summarize_text),
    ],
)
