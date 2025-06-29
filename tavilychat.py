import streamlit as st
import sqlite3
import os

from crewai import Agent, Task, Crew, Process
from crewai_tools import TavilySearch  # or others

st.write("sqlite3 version:", sqlite3.sqlite_version)


