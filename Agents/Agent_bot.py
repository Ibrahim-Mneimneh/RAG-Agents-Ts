from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")


class AgentState(TypedDict):
    message:List[HumanMessage]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=groq_api_key
)

def process(state:AgentState)->AgentState:
    response = llm.invoke(state["message"])
    print(f"Response: {response}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent=graph.compile()

user_input = input("Enter your message: ")
initial_state:AgentState = {"message":[HumanMessage(content=user_input)]}

agent.invoke(initial_state)