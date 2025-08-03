import streamlit as st
import requests
import openai
import re
import time
import logging
from functools import wraps
from typing import List, Dict, Tuple, Optional

# Configuration Constants
MAX_HISTORY_MESSAGES = 8
MAX_SEARCH_RESULTS = 3
REQUEST_TIMEOUT = 10
RATE_LIMIT_CALLS_PER_MINUTE = 10
MIN_NAME_LENGTH = 2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting decorator
def rate_limit(calls_per_minute: int = RATE_LIMIT_CALLS_PER_MINUTE):
    """Rate limiting decorator to prevent API abuse"""
    def decorator(func):
        func.last_called = 0
        func.call_count = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            min_interval = 60 / calls_per_minute
            
            if now - func.last_called < min_interval:
                wait_time = min_interval - (now - func.last_called)
                st.warning(f"Please wait {wait_time:.1f} seconds before making another request.")
                return [], "Rate limited - please wait before trying again."
            
            func.last_called = now
            func.call_count += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_input(text: str) -> bool:
    """Validate user input for basic safety and format"""
    if not text or not isinstance(text, str):
        return False
    
    # Check for reasonable length
    if len(text.strip()) == 0 or len(text) > 2000:
        return False
    
    # Basic sanitization - remove potential script injections
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True

def extract_name(text: str) -> Optional[str]:
    """
    Enhanced name extraction with better validation
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return None
    
    # Improved patterns with word boundaries
    patterns = [
        r'\bmy name is ([a-zA-Z\s]{2,30})\b',
        r'\bi am ([a-zA-Z\s]{2,30})\b', 
        r'\bi\'?m ([a-zA-Z\s]{2,30})\b',
        r'\bcall me ([a-zA-Z\s]{2,30})\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Take only the first word and validate
            first_name = name.split()[0].capitalize()
            if len(first_name) >= MIN_NAME_LENGTH and first_name.isalpha():
                return first_name
    
    return None

@rate_limit(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE)
def tavily_search(query: str) -> Tuple[List[Dict], str]:
    """
    Enhanced search function with comprehensive error handling
    Returns: (source_list, formatted_text)
    """
    if not validate_input(query):
        logger.warning(f"Invalid query rejected: {query[:50]}...")
        return [], "Invalid search query."
    
    url = "https://api.tavily.com/search"
    
    try:
        # Check if API key exists
        if 'tavily_key' not in st.secrets:
            st.error("Tavily API key not configured. Please check your secrets.")
            return [], "Search service not configured."
        
        headers = {
            "Authorization": f"Bearer {st.secrets['tavily_key']}",
            "Content-Type": "application/json",
            "User-Agent": "StreamlitChatbot/1.0"
        }
        
        data = {"query": query.strip()}
        
        # Make request with timeout
        response = requests.post(
            url, 
            json=data, 
            headers=headers, 
            timeout=REQUEST_TIMEOUT
        )
        
        response.raise_for_status()
        results = response.json()
        
        if not results or 'results' not in results:
            return [], "No search results found."
        
    except requests.exceptions.Timeout:
        logger.error("Tavily API timeout")
        st.error("Search request timed out. Please try again.")
        return [], "Search timed out."
    
    except requests.exceptions.HTTPError as e:
        logger.error(f"Tavily API HTTP error: {e}")
        if response.status_code == 401:
            st.error("API authentication failed. Please check your API key.")
        elif response.status_code == 429:
            st.error("Rate limit exceeded. Please wait before searching again.")
        else:
            st.error(f"Search service error: {response.status_code}")
        return [], f"Search failed with status {response.status_code}."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Tavily API request error: {e}")
        st.error("Network error occurred. Please check your connection.")
        return [], "Network error during search."
    
    except Exception as e:
        logger.error(f"Unexpected error in tavily_search: {e}")
        st.error("An unexpected error occurred during search.")
        return [], "Search encountered an unexpected error."
    
    # Process results
    sources = []
    formatted = []
    
    try:
        search_results = results.get("results", [])[:MAX_SEARCH_RESULTS]
        
        for i, item in enumerate(search_results, 1):
            # Safely extract fields with defaults
            snippet = item.get("content", "No content available")[:500]  # Limit snippet length
            url_link = item.get("url", "")
            title = item.get("title", f"Source {i}")[:100]  # Limit title length
            
            # Validate URL format
            if url_link and not url_link.startswith(('http://', 'https://')):
                url_link = ""
            
            source_info = {
                "id": i,
                "title": title,
                "url": url_link,
                "snippet": snippet
            }
            sources.append(source_info)
            
            # Format for display
            if url_link:
                formatted.append(f'> {snippet}\nSource [{i}]: {title} - {url_link}')
            else:
                formatted.append(f'> {snippet}\nSource [{i}]: {title}')
        
        formatted_text = "\n\n".join(formatted) if formatted else "No web results found."
        return sources, formatted_text
        
    except Exception as e:
        logger.error(f"Error processing search results: {e}")
        return [], "Error processing search results."

def add_footnotes_to_response(response_text: str, sources: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Process AI response to add footnote markers and track used sources
    Enhanced with better pattern matching and preserved hyperlinks
    """
    if not sources or not response_text:
        return response_text, []
    
    used_sources = []
    footnote_counter = 1
    
    # Pre-compile regex for efficiency
    source_pattern = re.compile(r'Source \[(\d+)\]', re.IGNORECASE)
    
    try:
        # Find all source references in the response
        matches = source_pattern.findall(response_text)
        unique_matches = list(dict.fromkeys(matches))  # Remove duplicates while preserving order
        
        for match in unique_matches:
            source_id = int(match)
            # Find corresponding source
            source = next((s for s in sources if s['id'] == source_id), None)
            
            if source and source not in used_sources:
                # Create clickable footnote with hyperlink if URL exists
                old_ref = f"Source [{source_id}]"
                
                if source.get('url'):
                    # Create a clickable footnote link
                    new_ref = f"[[^{footnote_counter}]]({source['url']})"
                else:
                    # Just a footnote marker if no URL
                    new_ref = f"[^{footnote_counter}]"
                
                # Replace all instances of this source reference
                response_text = response_text.replace(old_ref, new_ref)
                source['footnote_id'] = footnote_counter
                used_sources.append(source)
                footnote_counter += 1
        
        return response_text, used_sources
        
    except Exception as e:
        logger.error(f"Error processing footnotes: {e}")
        return response_text, []

def format_sources_list(used_sources: List[Dict]) -> str:
    """
    Format the sources list for display at the end of the response
    """
    if not used_sources:
        return ""
    
    try:
        sources_text = "\n\n---\n**Sources:**\n\n"
        for source in used_sources:
            footnote_id = source.get('footnote_id', source['id'])
            title = source.get('title', 'Unknown Source')
            url = source.get('url', '')
            
            if url:
                sources_text += f"[^{footnote_id}]: [{title}]({url})\n\n"
            else:
                sources_text += f"[^{footnote_id}]: {title}\n\n"
        
        return sources_text
        
    except Exception as e:
        logger.error(f"Error formatting sources: {e}")
        return "\n\n---\n**Sources:** Error formatting sources list\n\n"

def chat_with_gpt(name: Optional[str], history: List[Dict], user_input: str, 
                  sources: List[Dict], search_reply: str = "") -> Tuple[str, List[Dict]]:
    """
    Enhanced GPT chat function with better error handling
    """
    if not validate_input(user_input):
        return "I'm sorry, but I can't process that input. Please try rephrasing your question.", []
    
    try:
        # Check API key
        if 'api_key' not in st.secrets:
            st.error("OpenAI API key not configured.")
            return "AI service not available. Please contact support.", []
        
        # Build system message
        if name:
            system = (
                f"You are WebGPT, a helpful, friendly AI assistant chatting with {name}. "
                f"Address {name} personally when appropriate. "
                "You search the web for current information before responding. "
                "IMPORTANT: When referencing information from search results, you MUST use the exact format 'Source [X]' where X is the source number (1, 2, 3, etc.). "
                "These will be converted to clickable links automatically. Be sure to reference relevant sources in your response."
            )
        else:
            system = (
                "You are WebGPT, a helpful, friendly AI assistant. "
                "You search the web for current information before responding. "
                "IMPORTANT: When referencing information from search results, you MUST use the exact format 'Source [X]' where X is the source number (1, 2, 3, etc.). "
                "These will be converted to clickable links automatically. Be sure to reference relevant sources in your response."
            )

        # Build message history (limit to prevent token overflow)
        messages = [{"role": "system", "content": system}]
        
        # Add recent history
        recent_history = history[-MAX_HISTORY_MESSAGES:] if history else []
        for entry in recent_history:
            if entry.get("role") and entry.get("content"):
                messages.append(entry)
        
        # Add current user message with search context
        user_message = (
            f"User question: {user_input}\n\n"
            "Available web search results (use these for your response):\n"
            f"{search_reply}\n\n"
            "Instructions:\n"
            "1. Provide a comprehensive answer using the search results\n"
            "2. Reference specific information using 'Source [1]', 'Source [2]', etc.\n"
            "3. Make sure to cite the most relevant sources for key facts\n"
            "4. If search results don't contain enough information, mention this"
        )
        messages.append({"role": "user", "content": user_message})

        # Make API call
        client = openai.Client(api_key=st.secrets["api_key"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1500,  # Limit response length
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        if not raw_response:
            return "I apologize, but I couldn't generate a response. Please try again.", []
        
        # Process footnotes (this converts Source [X] to clickable links)
        footnoted_response, used_sources = add_footnotes_to_response(raw_response, sources)
        
        # Add traditional sources list at the end as backup
        sources_list = format_sources_list(used_sources)
        final_response = footnoted_response + sources_list
        
        return final_response, used_sources
        
    except openai.AuthenticationError:
        logger.error("OpenAI authentication failed")
        st.error("AI service authentication failed. Please check API key.")
        return "AI service unavailable due to authentication error.", []
    
    except openai.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        st.error("AI service rate limit exceeded. Please wait before trying again.")
        return "AI service temporarily unavailable due to high demand.", []
    
    except openai.APITimeoutError:
        logger.error("OpenAI API timeout")
        st.error("AI service timed out. Please try again.")
        return "AI service timed out. Please try your question again.", []
    
    except Exception as e:
        logger.error(f"Error in chat_with_gpt: {e}")
        st.error("An error occurred while processing your request.")
        return "I encountered an error processing your request. Please try again.", []

# Streamlit UI
def main():
    st.title("🔍 WebGPT - Your Personal Research Assistant")
    st.caption("Powered by web search and AI")

    # Initialize session state
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "all_sources" not in st.session_state:
        st.session_state.all_sources = []

    # Chat input with validation
    user_input = st.chat_input("Ask me anything or introduce yourself first!")

    if user_input:
        # Validate input
        if not validate_input(user_input):
            st.error("Please enter a valid question (1-2000 characters, no scripts).")
            return
        
        # Extract name if provided
        possible_name = extract_name(user_input)
        if possible_name:
            st.session_state.user_name = possible_name
            st.success(f"Nice to meet you, {possible_name}! 👋")

        # Search and get AI response
        with st.spinner("🔍 Searching the web..."):
            sources, web_snippets = tavily_search(user_input)
        
        if sources:  # Only proceed if search was successful
            with st.spinner("🤖 Generating response..."):
                answer, used_sources = chat_with_gpt(
                    st.session_state.user_name, 
                    st.session_state.message_history, 
                    user_input, 
                    sources,
                    search_reply=web_snippets
                )
            
            # Update conversation history
            st.session_state.message_history.append({"role": "user", "content": user_input})
            st.session_state.message_history.append({"role": "assistant", "content": answer})
            
            # Store sources (avoid duplicates)
            for source in used_sources:
                if not any(s['url'] == source['url'] for s in st.session_state.all_sources):
                    st.session_state.all_sources.append(source)

    # Display conversation history
    for msg in st.session_state.message_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Display sources sidebar
    if st.session_state.all_sources:
        with st.expander("📚 Sources Used in This Conversation"):
            for i, source in enumerate(st.session_state.all_sources, 1):
                if source.get('url'):
                    st.markdown(f"**{i}.** [{source['title']}]({source['url']})")
                else:
                    st.markdown(f"**{i}.** {source['title']}")

    # User info and controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.user_name:
            st.info(f"👋 Hi {st.session_state.user_name}! I remember your name.")
        else:
            st.info("💡 Tip: Tell me your name for a more personal conversation!")
    
    with col2:
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            st.session_state.message_history = []
            st.session_state.all_sources = []
            st.session_state.user_name = None
            st.rerun()

if __name__ == "__main__":
    main()
