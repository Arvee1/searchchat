import streamlit as st
import openai
import re
import time
import logging
import json
import subprocess
import tempfile
import os
from functools import wraps
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, urljoin

# Configuration Constants
MAX_HISTORY_MESSAGES = 8
MAX_SEARCH_RESULTS = 3
REQUEST_TIMEOUT = 30
RATE_LIMIT_CALLS_PER_MINUTE = 10
MIN_NAME_LENGTH = 2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP Server Configuration
MCP_SERVER_CONFIG = {
    "command": ["python", "-m", "mcp_web_scraper"],  # Adjust based on your MCP server implementation
    "args": [],
    "env": {}
}

class MCPWebScraper:
    """MCP Web Scraping Client"""
    
    def __init__(self):
        self.server_process = None
        self.session_id = None
        
    def start_server(self):
        """Start the MCP server process"""
        try:
            # Start the MCP server as a subprocess
            self.server_process = subprocess.Popen(
                MCP_SERVER_CONFIG["command"] + MCP_SERVER_CONFIG["args"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **MCP_SERVER_CONFIG["env"]}
            )
            
            # Initialize session
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "streamlit-chat",
                        "version": "1.0.0"
                    }
                }
            }
            
            self.server_process.stdin.write(json.dumps(init_request) + "\n")
            self.server_process.stdin.flush()
            
            # Read initialization response
            response = self.server_process.stdout.readline()
            if response:
                result = json.loads(response)
                logger.info(f"MCP server initialized: {result}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    def scrape_url(self, url: str) -> Dict:
        """Scrape content from a URL using MCP server"""
        try:
            if not self.server_process:
                if not self.start_server():
                    return {"error": "Failed to start MCP server"}
            
            request = {
                "jsonrpc": "2.0",
                "id": int(time.time()),
                "method": "tools/call",
                "params": {
                    "name": "scrape_url",
                    "arguments": {
                        "url": url,
                        "extract_text": True,
                        "extract_links": True,
                        "max_length": 2000
                    }
                }
            }
            
            self.server_process.stdin.write(json.dumps(request) + "\n")
            self.server_process.stdin.flush()
            
            # Read response with timeout
            response = self.server_process.stdout.readline()
            if response:
                result = json.loads(response)
                if "result" in result:
                    return result["result"]
                elif "error" in result:
                    return {"error": result["error"]["message"]}
            
            return {"error": "No response from MCP server"}
            
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            return {"error": f"Scraping failed: {str(e)}"}
    
    def search_and_scrape(self, query: str, urls: List[str]) -> List[Dict]:
        """Search and scrape multiple URLs"""
        results = []
        
        for i, url in enumerate(urls[:MAX_SEARCH_RESULTS], 1):
            try:
                # Validate URL
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    continue
                
                scraped_data = self.scrape_url(url)
                
                if "error" not in scraped_data:
                    # Extract relevant information
                    content = scraped_data.get("text", "")
                    title = scraped_data.get("title", f"Page {i}")
                    
                    # Limit content length
                    snippet = content[:500] + "..." if len(content) > 500 else content
                    
                    result = {
                        "id": i,
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "full_content": content
                    }
                    results.append(result)
                else:
                    logger.warning(f"Failed to scrape {url}: {scraped_data['error']}")
                    
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                continue
        
        return results
    
    def cleanup(self):
        """Clean up the MCP server process"""
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            except Exception as e:
                logger.error(f"Error cleaning up MCP server: {e}")

# Initialize MCP scraper (singleton pattern)
@st.cache_resource
def get_mcp_scraper():
    """Get or create MCP scraper instance"""
    return MCPWebScraper()

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
    """Enhanced name extraction with better validation"""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return None
    
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
            first_name = name.split()[0].capitalize()
            if len(first_name) >= MIN_NAME_LENGTH and first_name.isalpha():
                return first_name
    
    return None

def get_search_urls(query: str) -> List[str]:
    """
    Generate URLs to scrape based on the query.
    This is a simplified example - you might want to integrate with a search API
    or use a more sophisticated URL generation strategy.
    """
    # For demo purposes, using some common knowledge sites
    # In a real implementation, you'd want to:
    # 1. Use a search API to get relevant URLs
    # 2. Have a curated list of reliable sources
    # 3. Use domain-specific logic based on the query type
    
    base_urls = [
        f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        f"https://www.britannica.com/search?query={query.replace(' ', '+')}",
        f"https://news.google.com/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    ]
    
    return base_urls

@rate_limit(calls_per_minute=RATE_LIMIT_CALLS_PER_MINUTE)
def mcp_web_search(query: str) -> Tuple[List[Dict], str]:
    """
    Enhanced search function using MCP web scraping
    Returns: (source_list, formatted_text)
    """
    if not validate_input(query):
        logger.warning(f"Invalid query rejected: {query[:50]}...")
        return [], "Invalid search query."
    
    try:
        # Get MCP scraper instance
        scraper = get_mcp_scraper()
        
        # Get URLs to scrape
        urls_to_scrape = get_search_urls(query)
        
        # Scrape the URLs
        results = scraper.search_and_scrape(query, urls_to_scrape)
        
        if not results:
            return [], "No web results found for your query."
        
        # Format results for display
        formatted = []
        for result in results:
            snippet = result["snippet"]
            url = result["url"]
            title = result["title"]
            
            formatted.append(f'> {snippet}\nSource [{result["id"]}]: {title} - {url}')
        
        formatted_text = "\n\n".join(formatted)
        return results, formatted_text
        
    except Exception as e:
        logger.error(f"Error in mcp_web_search: {e}")
        st.error("An error occurred during web scraping.")
        return [], "Web scraping encountered an error."

def add_footnotes_to_response(response_text: str, sources: List[Dict]) -> Tuple[str, List[Dict]]:
    """Process AI response to add inline clickable links and track used sources"""
    if not sources or not response_text:
        return response_text, []
    
    used_sources = []
    
    try:
        patterns = [
            re.compile(r'Source \[(\d+)\]', re.IGNORECASE),
            re.compile(r'\[Source (\d+)\]', re.IGNORECASE),
            re.compile(r'source (\d+)', re.IGNORECASE),
        ]
        
        all_matches = []
        for pattern in patterns:
            matches = pattern.findall(response_text)
            for match in matches:
                all_matches.append((pattern, match))
        
        processed_ids = set()
        for pattern, match in all_matches:
            source_id = int(match)
            if source_id in processed_ids:
                continue
                
            source = next((s for s in sources if s['id'] == source_id), None)
            
            if source and source not in used_sources:
                old_refs = [
                    f"Source [{source_id}]",
                    f"[Source {source_id}]", 
                    f"source {source_id}"
                ]
                
                if source.get('url'):
                    title = source.get('title', f'Source {source_id}')[:50]
                    new_ref = f"[{title}]({source['url']})"
                else:
                    title = source.get('title', f'Source {source_id}')
                    new_ref = f"**{title}**"
                
                for old_ref in old_refs:
                    response_text = response_text.replace(old_ref, new_ref)
                
                used_sources.append(source)
                processed_ids.add(source_id)
        
        return response_text, used_sources
        
    except Exception as e:
        logger.error(f"Error processing footnotes: {e}")
        return response_text, []

def format_sources_list(used_sources: List[Dict]) -> str:
    """Format a comprehensive clickable sources list"""
    if not used_sources:
        return ""
    
    try:
        sources_text = "\n\n---\n\n### 📚 **Sources & References:**\n\n"
        
        for i, source in enumerate(used_sources, 1):
            title = source.get('title', f'Source {i}')
            url = source.get('url', '')
            snippet = source.get('snippet', '')[:150] + "..." if source.get('snippet') else ""
            
            if url:
                sources_text += f"**{i}. [{title}]({url})**\n"
                sources_text += f"🔗 `{url}`\n"
                if snippet:
                    sources_text += f"💭 *{snippet}*\n"
                sources_text += "\n"
            else:
                sources_text += f"**{i}. {title}**\n"
                if snippet:
                    sources_text += f"💭 *{snippet}*\n"
                sources_text += "\n"
        
        return sources_text
        
    except Exception as e:
        logger.error(f"Error formatting sources: {e}")
        return "\n\n---\n**Sources:** Error formatting sources list\n\n"

def chat_with_gpt(name: Optional[str], history: List[Dict], user_input: str, 
                  sources: List[Dict], search_reply: str = "") -> Tuple[str, List[Dict]]:
    """Enhanced GPT chat function using MCP scraped content"""
    if not validate_input(user_input):
        return "I'm sorry, but I can't process that input. Please try rephrasing your question.", []
    
    try:
        if 'api_key' not in st.secrets:
            st.error("OpenAI API key not configured.")
            return "AI service not available. Please contact support.", []
        
        # Build system message
        if name:
            system = (
                f"Wazzup!!!, I am Arvee your helpful assistant chatting with {name}. "
                f"Address {name} personally when appropriate. "
                "You use web scraping to get current information before responding. "
                "CRITICAL: When citing sources, you MUST use EXACTLY this format: 'Source [1]' or 'Source [2]' etc. "
                "Do NOT use '[Source 1]' or any other format. Use 'Source [1]' with a space before the bracket. "
                "Always cite relevant sources using this exact format to provide clickable links."
            )
        else:
            system = (
                "You are Arvee, a helpful, friendly AI assistant. "
                "You use web scraping to get current information before responding. "
                "CRITICAL: When citing sources, you MUST use EXACTLY this format: 'Source [1]' or 'Source [2]' etc. "
                "Do NOT use '[Source 1]' or any other format. Use 'Source [1]' with a space before the bracket. "
                "Always cite relevant sources using this exact format to provide clickable links."
            )

        messages = [{"role": "system", "content": system}]
        
        recent_history = history[-MAX_HISTORY_MESSAGES:] if history else []
        for entry in recent_history:
            if entry.get("role") and entry.get("content"):
                messages.append(entry)
        
        user_message = (
            f"User question: {user_input}\n\n"
            "Available web scraped content (numbered 1, 2, 3):\n"
            f"{search_reply}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "1. Answer the user's question using the scraped content above\n"
            "2. When citing information, use EXACTLY 'Source [1]' or 'Source [2]' format\n"
            "3. Example: 'According to the scraped data Source [1], the information shows...'\n"
            "4. DO NOT use '[Source 1]' - use 'Source [1]' with space before bracket\n"
            "5. Cite the most relevant sources for key facts and claims"
        )
        messages.append({"role": "user", "content": user_message})

        client = openai.Client(api_key=st.secrets["api_key"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        if not raw_response:
            return "I apologize, but I couldn't generate a response. Please try again.", []
        
        linked_response, used_sources = add_footnotes_to_response(raw_response, sources)
        sources_list = format_sources_list(used_sources)
        final_response = linked_response + sources_list
        
        return final_response, used_sources
        
    except openai.AuthenticationError:
        logger.error("OpenAI authentication failed")
        st.error("AI service authentication failed. Please check API key.")
        return "AI service unavailable due to authentication error.", []
    
    except openai.RateLimitError:
        logger.error("OpenAI rate limit exceeded")
        st.error("AI service rate limit exceeded. Please wait before trying again.")
        return "AI service temporarily unavailable due to high demand.", []
    
    except Exception as e:
        logger.error(f"Error in chat_with_gpt: {e}")
        st.error("An error occurred while processing your request.")
        return "I encountered an error processing your request. Please try again.", []

# Streamlit UI
def main():
    st.title("🔍 Wazzup!!! - I am your Personal Research Assistant")
    st.caption("Powered by MCP web scraping and LLM")

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
        if not validate_input(user_input):
            st.error("Please enter a valid question (1-2000 characters, no scripts).")
            return
        
        possible_name = extract_name(user_input)
        if possible_name:
            st.session_state.user_name = possible_name
            st.success(f"Nice to meet you, {possible_name}! 👋")

        # Search and scrape web content
        with st.spinner("🔍 Scraping web content..."):
            sources, web_content = mcp_web_search(user_input)
        
        if sources:
            with st.spinner("🤖 Generating response..."):
                answer, used_sources = chat_with_gpt(
                    st.session_state.user_name, 
                    st.session_state.message_history, 
                    user_input, 
                    sources,
                    search_reply=web_content
                )
            
            st.session_state.message_history.append({"role": "user", "content": user_input})
            st.session_state.message_history.append({"role": "assistant", "content": answer})
            
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

# Cleanup on app shutdown
import atexit

def cleanup_mcp_server():
    """Clean up MCP server on app shutdown"""
    try:
        scraper = get_mcp_scraper()
        scraper.cleanup()
    except:
        pass

atexit.register(cleanup_mcp_server)

if __name__ == "__main__":
    main()
