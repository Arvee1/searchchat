import streamlit as st
import os

from crewai import Agent, Task, Crew, Process
from crewai_tools import OpenAIChat
from crewai_tools_tools import TavilySearch

# Set API keys from secrets (edit for your project)
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["TAVILY_API_KEY"] = st.secrets["tavily"]["api_key"]

# Set up tool and agent
search = TavilySearch()
tools = [search]
agent = Agent(
    role="AI Assistant",
    goal="Help the user with their questions wisely and kindly.",
    tools=tools,
    backstory="You are a helpful, clever AI assistant.",
    llm=OpenAIChat(model="gpt-4o"),
)

st.title("CrewAI Streamlit Chatbot (Modern)")

# Store memory in session state
if "history" not in st.session_state:
    st.session_state.history = []

user_text = st.text_input("Say something:")

if user_text:
    st.session_state.history.append(("User", user_text))
    # Build history/context for the agent:
    full_history = "\n".join(f"{role}: {msg}" for role, msg in st.session_state.history)
    prompt = f"""Here is the conversation so far:

{full_history}

As Assistant, respond helpfully and completely to the latest user input."""

    task = Task(description=prompt, agent=agent)
    # Crew: (agent(s), task(s), process type)
    result = Crew([agent], [task], Process.sequential).run()
    st.session_state.history.append(("Assistant", result))
    st.write(result)

if st.session_state.history:
    st.markdown("### Chat Transcript")
    for role, msg in st.session_state.history:
        st.markdown(f"**{role}:** {msg}")
