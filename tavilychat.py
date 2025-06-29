import streamlit as st
import requests
import openai
import os

openai.api_key = st.secrets["api_key"]

def serper_search(query):
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": st.secrets["tavily_key"]
    }
    data = {"q": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    # We'll grab the top organic results for brevity
    snippets = [item["snippet"] for item in results.get("organic", [])[:3]]
    return "\n".join(snippets) if snippets else "No web results found."

def chat_with_gpt(prompt, search_reply=""):
    system = (
        "You are a helpful and concise assistant. "
        "If relevant, incorporate these recent web search results below into your answer."
    )
    full_prompt = f"{prompt}\n\nWeb results:\n{search_reply}"
    chat_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": full_prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or gpt-4-turbo or gpt-3.5-turbo
        messages=chat_messages,
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"].strip()

st.title("OpenAI + Serper Web Search Chat (no CrewAI needed)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask me anything (with web search!)")

if user_input:
    st.session_state.history.append(("User", user_input))
    web_snippets = serper_search(user_input)
    answer = chat_with_gpt(user_input, search_reply=web_snippets)
    st.session_state.history.append(("Assistant", answer))
    st.write(answer)

if st.session_state.history:
    st.markdown("---")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
