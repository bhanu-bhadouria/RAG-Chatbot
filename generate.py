from openai import OpenAI
import os
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def generate_answer(query, context, chat_history=""):
    prompt = f"""
    You are a helpful assistant.

    Use the conversation history and the retrieved context to answer the user's question.

    If the answer is not present in the context, say "I don't know."

    --- Conversation History ---
    {chat_history}

    --- Context ---
    {context}

    --- Question ---
    {query}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
                ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating answer: {e}")
        return "⚠️ Sorry, I couldn't generate an answer right now."