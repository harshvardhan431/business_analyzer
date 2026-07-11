import os
import time
from groq import Groq
from rag.retreiver import retrieve

SYSTEM_PROMPT = """You are an expert AI business consultant and analyst. You help users with:
- Product promotion strategy
- Marketing plan creation
- Ad copy and pitch writing
- Business growth analysis and prediction
- General business, finance, and market questions

If the user has uploaded product documents, prioritize that information and mention which 
document it came from. If no relevant documents are found, answer from your general knowledge.
If you know facts about the user or their business from memory, use them to personalize answers.
Always be specific, practical, and direct."""


def build_messages(user_message: str, context_chunks: list[dict], history: list, memory_context: str = "") -> list:

    if context_chunks:
        context_parts = []
        for chunk in context_chunks:
            context_parts.append(
                f"[From {chunk['source']}, relevance {1 - chunk['distance']:.0%}]:\n{chunk['text']}"
            )
        context_str   = "\n\n".join(context_parts)
        context_block = f"\n\nRelevant content from uploaded documents:\n{context_str}\n"
    else:
        context_block = ""

    memory_block = f"\n\n{memory_context}\n" if memory_context else ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for log in history[:-1]:
        role = "user" if log.role == "user" else "assistant"
        messages.append({"role": role, "content": log.message})

    enriched = f"{memory_block}{context_block}\nUser: {user_message}" if (memory_block or context_block) else user_message
    messages.append({"role": "user", "content": enriched})

    return messages


def get_response(user_message: str, history: list, memory_context: str = "") -> tuple[str, str]:
    context_chunks = retrieve(user_message, top_k=4)

    if context_chunks:
        sources      = list(set(c["source"] for c in context_chunks))
        source_label = "📄 " + ", ".join(sources)
    else:
        source_label = "🤖 AI general knowledge"

    messages = build_messages(user_message, context_chunks, history, memory_context)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            return response.choices[0].message.content, source_label
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3)
                continue
            return f"Sorry, something went wrong: {str(e)}", source_label

    return "AI is busy. Please try again.", source_label