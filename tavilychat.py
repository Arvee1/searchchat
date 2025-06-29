from crewai import Agent, Task, Crew, Process
from crewai_tools import OpenAIChat
from crewai_tools_tools import TavilySearch  # or SerperSearch, etc.

import os
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["TAVILY_API_KEY"] = st.secrets["tavily"]["api_key"]

# Define your tools
search = TavilySearch()
tools = [search]

# Create your Agent
agent = Agent(
    role="AI Assistant",
    goal="Help the user with their questions",
    tools=tools,
    backstory="You are a helpful AI assistant.",
    llm=OpenAIChat(model="gpt-4o"),
)

st.title("CrewAI Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

user_text = st.text_input("Say something:")

if user_text:
    st.session_state.history.append(("User", user_text))
    # Gather the chat history and present as context to agent
    chat_history = ""
    for speaker, msg in st.session_state.history:
        chat_history += f"{speaker}: {msg}\n"
    task = Task(
        description=f"The conversation so far:\n{chat_history}\n"
                    f"Respond to the user's latest input as a helpful assistant.",
        agent=agent
    )
    result = Crew([agent], [task], Process.sequential).run()
    st.session_state.history.append(("Assistant", result))
    st.write(result)

if st.session_state.history:
    st.markdown("### Conversation so far")
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")
