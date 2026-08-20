import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core import tools
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage 
from langchain_core.messages import ToolMessage 
from langchain_core.messages import SystemMessage 
from langgraph.graph.message import add_messages
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-safeguard-20b",
    temperature=0,
    groq_api_key=groq_api_key
)

document_content =""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content:str)-> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document content updated to:\n\n{document_content}\n\n"

@tool
def save(filename:str)-> str:
    """Saves the current document to a text file and finish the process.
    Args:
        filename (str): Name of the text file
    """
    if not filename.endswith(".txt"):
        filename=f"{filename}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)
            print(f"\nDocument saved to {filename}\n")
    except Exception as e:
        print(f"\nError saving document: {e}\n")
        return f"Error saving document: {e}"


tools = [update, save]
model =llm.bind_tools(tools)

    
def agent(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """
    )

    if not state["messages"]:
        default_message = "Im ready to help you draft your document. Please provide the content or instructions."
        user_message = HumanMessage(content=default_message)
    else:
        user_input = input("What would you like to do with the document?")
        print(f"\nUser: {user_input}\n")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]+ [user_message])

    response = model.invoke(all_messages)
    print(20*"-")
    print(f"\nDrafter: {response.content}\n")
    print(20*"-")
    if hasattr(response, "tool_calls"):
        print(f" Using tools:\n")
        for tc in response.tool_calls:
            print(f"{tc['name']}\n")
    return {"messages":list(state["messages"])+[user_message,response]}

def should_continue(state:AgentState) -> str:
    messages=state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
    
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", agent)
graph.add_node("tools",ToolNode(name="tools", tools=tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})

app= graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()
