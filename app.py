import os
import streamlit as st
import datetime
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import PyPDF2
import pandas as pd
from docx import Document

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API key is available
if not API_KEY:
    st.error("OPENAI_API_KEY not found. Please ensure it is set in the .env file.")
    st.stop()

# Wikipedia API for external knowledge
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Agents
history_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
english_llm = ChatOpenAI(temperature=0, api_key=API_KEY, model="gpt-4o-mini")
history_agent = create_react_agent(history_llm, [wikipedia_tool])
english_agent = create_react_agent(english_llm, [wikipedia_tool])

# UI Setup
st.set_page_config(page_title="Chat with Expert", layout="wide")
st.sidebar.subheader("Select an Expert")
agent_choice = st.sidebar.radio("Choose an Expert:", ["History Expert", "English Expert"])
st.sidebar.subheader("Upload a File")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, TXT, CSV", type=["pdf", "docx", "txt", "csv"])

# File text extraction
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        return file.read().decode("utf-8", errors="replace")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/csv":
        return pd.read_csv(file).to_string()
    return "Unsupported file format."

# Session state management
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if uploaded_file:
    text = extract_text_from_file(uploaded_file)
    if text != "Unsupported file format.":
        st.session_state.document_text = text
        st.sidebar.success("File uploaded and processed successfully!")
    else:
        st.sidebar.error("Could not process the uploaded file.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interaction
st.title("💬 Chat with Expert")
st.subheader(f"Chat with the {agent_choice}")
user_input = st.text_input("Ask a question:")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.write("Thinking...")

    prompt = user_input
    if st.session_state.document_text:
        prompt = f"Based on this document, answer the question: {user_input}\n\nDocument:\n{st.session_state.document_text[:2000]}"

    agent = history_agent if agent_choice == "History Expert" else english_agent
    response = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    reply = response["messages"][-1].content

    st.session_state.chat_history.append(("bot", reply))

# Display chat
for sender, message in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"<div style='text-align: right; background-color: #4CAF50; color: white; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #f1f1f1; color: black; padding: 10px; border-radius: 10px; margin: 5px 0;'>{message}</div>", unsafe_allow_html=True)

# End session function
def kill_session():
    end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log = f"Session ended at {end_time}\nTotal messages exchanged: {len(st.session_state.chat_history)}"
    with open("session_log.txt", "a", encoding="utf-8", errors="replace") as log_file:
        log_file.write(log + "\n")

    st.session_state.chat_history.clear()
    st.session_state.document_text = ""
    st.sidebar.success("Session ended. API usage stopped.")
    st.stop()

st.sidebar.button("End Session", on_click=kill_session)
