import streamlit as st
import os
from crewai_tools.tools.llm.openai_chat import OpenAIChat
from crewai_tools import SerperSearch

from crewai import Agent, Task, Crew, Process
# from crewai_tools import OpenAIChat, SerperSearch   # <-- this is the correct import

# Set environment vars from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
os.environ["SERPER_API_KEY"] = st.secrets["api_key"]

search = SerperSearch()
tools = [search]
agent = Agent(
    role="AI Assistant",
    goal="Help the user with information from web search as needed.",
    tools=tools,
    backstory="You are a helpful assistant capable of searching the web.",
    llm=OpenAIChat(model="gpt-4o"),
)

st.title("CrewAI Chatbot with Serper Web Search")

if "history" not in st.session_state:
    st.session_state.history = []

user_text = st.text_input("You:", key="input")

if user_text:
    st.session_state.history.append(("User", user_text))
    # Build conversation context
    convo = "\n".join([f"{speaker}: {msg}" for speaker, msg in st.session_state.history])
    prompt = (f"Here is the conversation so far:\n{convo}\n"
              "Respond to the user's latest input, using search if needed.")

    task = Task(description=prompt, agent=agent)
    result = Crew([agent], [task], Process.sequential).run()
    st.session_state.history.append(("Assistant", result))
    st.write(result)

if st.session_state.history:
    st.markdown("### Conversation so far")
    for speaker, msg in st.session_state.history:
        st.markdown(f"**{speaker}:** {msg}")
