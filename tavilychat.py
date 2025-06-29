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
    # Format web results with snippets and sources if available
    formatted = []
    for item in results.get("results", [])[:3]:
        snippet = item["content"]
        link = item.get("url", "")
        if link:
            formatted.append(f'> {snippet}\nSource: {link}')
        else:
            formatted.append(f'> {snippet}')
    return "\n\n".join(formatted) if formatted else "No web results found."

def chat_with_gpt(history, user_input, search_reply=""):
    system = (
        "You are WebGPT, an AI assistant that always searches the web for up-to-date information before answering. "
        "If relevant, integrate quoted web results below (with sources) in your answer. "
        "If there are no relevant search results, say so."
    )

    # Prepare chat history (except for the system message), up to 8 exchanges
    messages = [{"role": "system", "content": system}]
    for entry in history[-8:]:
        messages.append(entry)
    # Add user turn with fresh search
    user_message = (
        f"{user_input}\n\n"
        "Web search results (quotes and sources):\n"
        f"{search_reply}\n\n"
        "When you answer, reference relevant info and cite sources where possible. If you can't find an answer, let the user know."
    )
    messages.append({"role": "user", "content": user_message})

    client = openai.Client(api_key=st.secrets["api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

st.title("WebGPT – Friendly AI with Real-Time Web Search")

if "message_history" not in st.session_state:
    st.session_state.message_history = []

# --- Improved chat UI with chat_input and nice bubbles (for Streamlit >=1.29) ---
user_input = st.chat_input("Ask anything (I will search the web for you!)")

if user_input:
    with st.spinner("Searching the web..."):
        web_snippets = tavily_search(user_input)
    with st.spinner("Talking to AI..."):
        answer = chat_with_gpt(st.session_state.message_history, user_input, search_reply=web_snippets)
    # Store both user and assistant turn
    st.session_state.message_history.append({"role": "user", "content": user_input})
    st.session_state.message_history.append({"role": "assistant", "content": answer})

# Display the conversation in chat bubbles
for msg in st.session_state.message_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

st.info("This AI always searches the web for your query, and references sources when possible.")
