import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import OpenAIChat
from crewai_tools_tools import TavilySearch

# Set up API keys from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
os.environ["TAVILY_API_KEY"] = st.secrets["tavily"]["api_key"]

search = TavilySearch()
tools = [search]
agent = Agent(
    role="AI Assistant",
    goal="Help user with their questions, using chat history.",
    tools=tools,
    backstory="You are a kind, clever assistant.",
    llm=OpenAIChat(model="gpt-4o"),
)

st.title("CrewAI Chat with Memory")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("Say something:")

if user_input:
    st.session_state.history.append(("User", user_input))

    # Prepare the context for the agent: concatenate history
    history_text = ""
    for speaker, text in st.session_state.history:
        prefix = "User:" if speaker == "User" else "Assistant:"
        history_text += f"{prefix} {text}\n"

    # Task prompt includes the chat history so far, and the current user message
    full_prompt = (
        f"Here is the conversation so far:\n{history_text}\n"
        f"As the AI Assistant, answer the latest user message as helpfully as possible."
    )

    task = Task(
        description=full_prompt,
        agent=agent
    )

    # Run the CrewAI agent
    result = Crew([agent], [task], Process.sequential).run()
    st.session_state.history.append(("Assistant", result))
    st.write(result)

# Display conversation history
if st.session_state.history:
    st.markdown("### Conversation")
    for speaker, text in st.session_state.history:
        # You can prettify this with markdown or chat bubbles, if desired
        st.markdown(f"**{speaker}:** {text}")
