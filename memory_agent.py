import os
import json
from groq import Groq

EXTRACT_PROMPT = """You are a memory extraction assistant. 
Given a conversation message, extract any important long-term facts about the user or their business.

Extract things like:
- Product name or type
- Business name
- Target audience
- Goals or problems they mentioned
- Budget or constraints
- Industry or niche
- Any personal preferences they stated

Return a JSON object with key-value pairs. Keys should be short snake_case strings.
If nothing worth remembering, return empty JSON {}.

Examples:
User says "I sell handmade candles targeting women aged 25-40"
Return: {"product": "handmade candles", "target_audience": "women aged 25-40"}

User says "what is machine learning"
Return: {}

Only return the JSON object, nothing else."""


def extract_memories(message: str) -> dict:
    try:
        client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": f"{EXTRACT_PROMPT}\n\nUser message: {message}"}
            ]
        )
        text = response.choices[0].message.content.strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)
    except Exception:
        return {}


def save_memories(user_id: int, memories: dict, db, UserMemory):
    for key, value in memories.items():
        existing = UserMemory.query.filter_by(user_id=user_id, key=key).first()
        if existing:
            existing.value = str(value)
        else:
            db.session.add(UserMemory(
                user_id=user_id,
                key=key,
                value=str(value)
            ))
    db.session.commit()


def get_memory_context(user_id: int, UserMemory) -> str:
    memories = UserMemory.query.filter_by(user_id=user_id).all()
    if not memories:
        return ""

    lines = [f"- {m.key.replace('_', ' ').title()}: {m.value}" for m in memories]
    return "What I know about this user:\n" + "\n".join(lines)