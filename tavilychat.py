import streamlit as st
import requests
import openai
import re

# --- Helper: Extract name from user input (simple patterns) ---
def extract_name(text):
    # Simple regex, can be improved for more patterns
    pattern = re.compile(r"my name is ([a-zA-Z ]+)|i am ([a-zA-Z ]+)|i'm ([a-zA-Z ]+)|call me ([a-zA-Z ]+)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        # Filter out None values, get first present group
        name = next((g for g in match.groups() if g), None)
        # Clean up extra words/common patterns
        if name:
            name = name.strip().split()[0].capitalize()
        return name
    return None

def tavily_search(query):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['tavily_key']}"
    }
    data = {"query": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    formatted = []
    for item in results.get("results", [])[:3]:
        snippet = item["content"]
        link = item.get("url", "")
        if link:
            formatted.append(f'> {snippet}\nSource: {link}')
        else:
            formatted.append(f'> {snippet}')
    return "\n\n".join(formatted) if formatted else "No web results found."

def chat_with_gpt(name, history, user_input, search_reply=""):
    if name:
        system = (
            f"You are WebGPT, a helpful, friendly AI assistant in a chat with {name}. "
            f"Always talk to {name} in a personal, welcoming tone and, when possible, use their name in your response. "
            "You always search the web for up-to-date information before answering. "
            "If relevant, integrate and cite web results (with sources) in your answer."
        )
    else:
        system = (
            "You are WebGPT, a helpful, friendly AI assistant. "
            "Always talk to the user in a personal tone, and if they tell you their name, use it in chat. "
            "You always search the web for up-to-date information before answering. "
            "If relevant, integrate and cite web results (with sources) in your answer."
        )

    messages = [{"role": "system", "content": system}]
    for entry in history[-8:]:
        messages.append(entry)
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

st.title("Wazzup!!! – It is me Arvee, your personal researcher!")

# --- Memory initialization ---
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None

# --- Chat input ---
user_input = st.chat_input("Ask me anything or introduce yourself first!")

if user_input:
    # Detect and save name if given
    possible_name = extract_name(user_input)
    if possible_name:
        st.session_state.user_name = possible_name

    with st.spinner("Searching the web..."):
        web_snippets = tavily_search(user_input)
    with st.spinner("Asking the AI to make sense of it..."):
        answer = chat_with_gpt(st.session_state.user_name, st.session_state.message_history, user_input, search_reply=web_snippets)
    st.session_state.message_history.append({"role": "user", "content": user_input})
    st.session_state.message_history.append({"role": "assistant", "content": answer})

# --- Display conversation history with personalization ---
for msg in st.session_state.message_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if st.session_state.user_name:
    st.info(f"👋 Hi {st.session_state.user_name}! The AI knows your name and will use it in the conversation.")
else:
    st.info("Tip: If you tell me your name, I'll remember it to make our conversation more personal!")
