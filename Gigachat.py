import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START

load_dotenv()

def create_gigachat_client():
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        verify_ssl_certs=False,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        timeout=30
    )

llm = create_gigachat_client()
test_response = llm.invoke([HumanMessage(content="Привет! Ты работаешь?")])
print("🤖 GigaChat ответил:", test_response.content)



