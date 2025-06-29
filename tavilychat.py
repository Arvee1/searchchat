import os
import streamlit as st
import sqlite3 as sql
import pandas as pd 
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display 
from langchain_core.messages import HumanMessage
from crewai_tools import create_react_agent, init_chat_model
from crewai_tools_tools import TavilySearch

# Set API keys from session state 
# Read your OpenAI API key from secrets and set as env var before model init
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
tavily_key = st.secrets["tavily_key"]

# Session state setup for chat memory
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "history" not in st.session_state:
    st.session_state.history = []

# Model, search, agent setup (no need to recreate every rerun, could cache if desired)
model = init_chat_model("openai:gpt-4o")
search = TavilySearch(tavily_api_key=tavily_key, max_results=2)
tools = [search]
agent_executor = create_react_agent(model, tools)
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
for msg in st.session_state.history:
    st.write(f"{msg['role']}: {msg['content']}")
