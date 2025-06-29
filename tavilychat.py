import streamlit as st
import requests
import openai

openai.api_key = st.secrets["api_key"]

def tavily_search(query):
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['tavily_key']}"
    }
    data = {"query": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    # Get up to 3 snippet results for context
    snippets = [item["content"] for item in results.get("results", [])[:3]]
    return "\n\n".join(snippets) if snippets else "No web results found."

def chat_with_gpt(prompt, search_reply=""):
    system = (
        "You are a helpful and concise assistant. "
        "When possible, use the provided web search snippets to answer."
    )
    user_message = f"{prompt}\n\nWeb results:\n{search_reply}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message}
    ]
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-4o",  # or gpt-3.5-turbo/gpt-4-turbo
    #     messages=messages,
    #     temperature=0.3,
    # )
    return response["choices"][0]["message"]["content"].strip()
    client = openai.Client(api_key=st.secrets["openai"]["api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    result = response.choices[0].message.content.strip()
    

st.title("ChatGPT with Tavily Web Search (No CrewAI)")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask anything (web-backed!)")

if user_input:
    st.session_state.history.append(("User", user_input))
    web_snippets = tavily_search(user_input)
    answer = chat_with_gpt(user_input, search_reply=web_snippets)
    st.session_state.history.append(("Assistant", answer))
    st.write(answer)

# Display entire conversation
if st.session_state.history:
    st.markdown("---")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
