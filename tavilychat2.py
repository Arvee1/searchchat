import streamlit as st
import requests
import openai
import re
from typing import List, Dict, Tuple

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

def tavily_search(query) -> Tuple[List[Dict], str]:
    """
    Enhanced search function that returns structured source data and formatted text
    Returns: (source_list, formatted_text)
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['tavily_key']}"
    }
    data = {"query": query}
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    results = response.json()
    
    sources = []
    formatted = []
    
    for i, item in enumerate(results.get("results", [])[:3], 1):
        snippet = item["content"]
        url_link = item.get("url", "")
        title = item.get("title", f"Source {i}")
        
        # Store source information
        source_info = {
            "id": i,
            "title": title,
            "url": url_link,
            "snippet": snippet
        }
        sources.append(source_info)
        
        # Format for display with source reference
        if url_link:
            formatted.append(f'> {snippet}\nSource [{i}]: {title} - {url_link}')
        else:
            formatted.append(f'> {snippet}\nSource [{i}]: {title}')
    
    formatted_text = "\n\n".join(formatted) if formatted else "No web results found."
    return sources, formatted_text

def add_footnotes_to_response(response_text: str, sources: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Process AI response to add footnote markers and track used sources
    """
    if not sources:
        return response_text, []
    
    used_sources = []
    footnote_counter = 1
    
    # Create a mapping of source references in the response
    for source in sources:
        source_patterns = [
            f"Source [{source['id']}]",
            f"source {source['id']}",
            source['title'][:30] if source['title'] else "",
            source['url'].split('/')[-1] if source['url'] else ""
        ]
        
        # Check if any source pattern appears in the response
        for pattern in source_patterns:
            if pattern and pattern.lower() in response_text.lower():
                if source not in used_sources:
                    # Replace the first occurrence with a footnote
                    footnote_marker = f"[^{footnote_counter}]"
                    response_text = response_text.replace(
                        f"Source [{source['id']}]", 
                        footnote_marker, 
                        1
                    )
                    source['footnote_id'] = footnote_counter
                    used_sources.append(source)
                    footnote_counter += 1
                break
    
    return response_text, used_sources

def format_sources_list(used_sources: List[Dict]) -> str:
    """
    Format the sources list for display at the end of the response
    """
    if not used_sources:
        return ""
    
    sources_text = "\n\n---\n**Sources:**\n\n"
    for source in used_sources:
        footnote_id = source.get('footnote_id', source['id'])
        title = source['title']
        url = source['url']
        
        if url:
            sources_text += f"[^{footnote_id}]: [{title}]({url})\n\n"
        else:
            sources_text += f"[^{footnote_id}]: {title}\n\n"
    
    return sources_text

def chat_with_gpt(name, history, user_input, sources, search_reply=""):
    if name:
        system = (
            f"You are WebGPT, a helpful, friendly AI assistant in a chat with {name}. "
            f"Always talk to {name} in a personal, welcoming tone and, when possible, use their name in your response. "
            "You always search the web for up-to-date information before answering. "
            "When referencing information from web search results, use the format 'Source [X]' where X is the source number. "
            "Integrate relevant information naturally into your response and cite sources appropriately."
        )
    else:
        system = (
            "You are WebGPT, a helpful, friendly AI assistant. "
            "Always talk to the user in a personal tone, and if they tell you their name, use it in chat. "
            "You always search the web for up-to-date information before answering. "
            "When referencing information from web search results, use the format 'Source [X]' where X is the source number. "
            "Integrate relevant information naturally into your response and cite sources appropriately."
        )

    messages = [{"role": "system", "content": system}]
    for entry in history[-8:]:
        messages.append(entry)
    
    user_message = (
        f"{user_input}\n\n"
        "Web search results (numbered sources for reference):\n"
        f"{search_reply}\n\n"
        "When you answer, reference relevant info using 'Source [number]' format. "
        "If you can't find an answer, let the user know."
    )
    messages.append({"role": "user", "content": user_message})

    client = openai.Client(api_key=st.secrets["api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    
    raw_response = response.choices[0].message.content.strip()
    
    # Process the response to add footnotes and get used sources
    footnoted_response, used_sources = add_footnotes_to_response(raw_response, sources)
    
    # Add sources list at the end
    sources_list = format_sources_list(used_sources)
    final_response = footnoted_response + sources_list
    
    return final_response, used_sources

st.title("Wazzup!!! – It is me Arvee, your personal researcher!")

# --- Memory initialization ---
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "all_sources" not in st.session_state:
    st.session_state.all_sources = []

# --- Chat input ---
user_input = st.chat_input("Ask me anything or introduce yourself first!")

if user_input:
    # Detect and save name if given
    possible_name = extract_name(user_input)
    if possible_name:
        st.session_state.user_name = possible_name

    with st.spinner("Searching the web..."):
        sources, web_snippets = tavily_search(user_input)
    
    with st.spinner("Asking the AI to make sense of it..."):
        answer, used_sources = chat_with_gpt(
            st.session_state.user_name, 
            st.session_state.message_history, 
            user_input, 
            sources,
            search_reply=web_snippets
        )
    
    # Store sources for this conversation
    st.session_state.all_sources.extend(used_sources)
    
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

# --- Display comprehensive sources list ---
if st.session_state.all_sources:
    with st.expander("📚 All Sources Referenced in This Conversation"):
        unique_sources = []
        seen_urls = set()
        
        for source in st.session_state.all_sources:
            if source['url'] not in seen_urls:
                unique_sources.append(source)
                seen_urls.add(source['url'])
        
        for i, source in enumerate(unique_sources, 1):
            if source['url']:
                st.markdown(f"**{i}.** [{source['title']}]({source['url']})")
            else:
                st.markdown(f"**{i}.** {source['title']}")

# --- Personalization info ---
if st.session_state.user_name:
    st.info(f"👋 Hi {st.session_state.user_name}! The AI knows your name and will use it in the conversation.")
else:
    st.info("Tip: If you tell me your name, I'll remember it to make our conversation more personal!")

# --- Clear conversation button ---
if st.button("🗑️ Clear Conversation"):
    st.session_state.message_history = []
    st.session_state.all_sources = []
    st.rerun()
