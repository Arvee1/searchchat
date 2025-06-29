import streamlit as st
from crewai_tools import TavilySearch, MemorySaver, create_react_agent, init_chat_model

# API key setup (do this your preferred way)
tavily_key = "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"

# Session state setup for chat memory
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "history" not in st.session_state:
    st.session_state.history = []

# Model, search, agent setup (no need to recreate every rerun, could cache if desired)
model = init_chat_model("openai:gpt-4o")
search = TavilySearch(tavily_api_key=tavily_key, max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools, checkpointer=st.session_state.memory)
config = {"configurable": {"thread_id": "abc123"}}

st.title("Tavily Chatbot (Streamlit Demo)")

# Chat UI, using chat_input (Streamlit 1.31+), fallback to text_input if needed
user_text = st.chat_input("Type your message...")

if user_text:  # When user sends a message
    st.session_state.history.append({"role": "user", "content": user_text})

    input_message = {
        "role": "user",
        "content": user_text,
    }

    # Display user message in chat
    with st.chat_message("user"):
        st.write(user_text)

    # Generate and display agent response
    full_response = ""  # If you want to collect all responses
    with st.chat_message("assistant"):
        for step in agent_executor.stream(
            {"messages": [input_message]}, config, stream_mode="values"
        ):
            # Print response (adapted for Streamlit)
            msg_str = str(step["messages"][-1])  # or .content or customize as needed
            st.write(msg_str)
            full_response += msg_str

    st.session_state.history.append({"role": "assistant", "content": full_response})

# Optionally, show chat history (for debugging)
# for msg in st.session_state.history:
#     st.write(f"{msg['role']}: {msg['content']}")
