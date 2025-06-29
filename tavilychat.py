import streamlit as st
import requests
import openai

def tavily_search(query):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['tavily_key']}"
    }
    data = {"query": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    snippets = [item["content"] for item in results.get("results", [])[:3]]
    return "\n\n".join(snippets) if snippets else "No web results found."

def chat_with_gpt(history, current_input, search_reply=""):
    system = (
        "You are a helpful and concise assistant. "
        "When possible, use the provided web search snippets to answer."
    )
    messages = [{"role": "system", "content": system}]
    # Add chat history; only keep the most recent 8 exchanges for context
    for entry in history[-8:]:
        messages.append(entry)
    # Add the current user message, with the latest web results
    user_message = f"{current_input}\n\nWeb results:\n{search_reply}"
    messages.append({"role": "user", "content": user_message})

    client = openai.Client(api_key=st.secrets["api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

st.title("ChatGPT + Tavily Web Search (with Memory)")

# Initialize history: a list of {"role": ..., "content": ...}
if "message_history" not in st.session_state:
    st.session_state.message_history = []

user_input = st.text_input("Ask anything (web-backed!)")

if user_input:
    with st.spinner("Searching the web..."):
        web_snippets = tavily_search(user_input)
    with st.spinner("Getting answer from GPT..."):
        answer = chat_with_gpt(st.session_state.message_history, user_input, search_reply=web_snippets)
    # Save both turns to chat memory
    st.session_state.message_history.append({"role": "user", "content": user_input})
    st.session_state.message_history.append({"role": "assistant", "content": answer})
    st.write(answer)

# Display full conversation history
if st.session_state.message_history:
    st.markdown("---")
    for msg in st.session_state.message_history:
        if msg["role"] == "user":
            st.markdown(f"**User:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Assistant:** {msg['content']}")
