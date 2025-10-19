from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from uipath.models import CreateAction
from uipath_langchain.chat import UiPathChat
from langchain_core.messages import SystemMessage, HumanMessage
from uipath_langchain.retrievers import ContextGroundingRetriever
from uipath import UiPath
from typing import Dict, Any
from dotenv import load_dotenv
from datetime import datetime
from contextlib import asynccontextmanager
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import json, os, logging, ast

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Use UiPathChat for making LLM calls
llm = UiPathChat(model="gpt-4o-2024-08-06")

uipath_client = UiPath()

# ---------------- MCP Server Configuration ----------------
@asynccontextmanager
async def get_mcp_session():
    """MCP session management"""
    MCP_SERVER_URL = os.getenv("UIPATH_MCP_SERVER_URL")
    print(MCP_SERVER_URL)
    if hasattr(uipath_client, 'api_client'):
                if hasattr(uipath_client.api_client, 'default_headers'):
                    auth_header = uipath_client.api_client.default_headers.get('Authorization', '')
                    if auth_header.startswith('Bearer '):
                        UIPATH_ACCESS_TOKEN = auth_header.replace('Bearer ', '')
                        logging.info("Retrieved token from UiPath API client")
    
    async with streamablehttp_client(
        url=MCP_SERVER_URL,
        headers={"Authorization": f"Bearer {UIPATH_ACCESS_TOKEN}"} if UIPATH_ACCESS_TOKEN else {},
        timeout=60,
    ) as (read, write, session_id_callback):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def get_mcp_tools():
    """Load MCP tools for use with agents"""
    print("Loading MCP tools...")
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        return tools
    
# Initialize Context Grounding for company policy
context_grounding = ContextGroundingRetriever(
    index_name="AssistmAIgic_Index",
    folder_path="Shared",
    number_of_results=1
    )

# ---------------- State ----------------
class GraphState(BaseModel):
    """Enhanced state to track the complete assistmAIgic workflow"""
    message_id: str | None = None
    email_language: str | None = None
    email_body: str | None = None
    email_subject: str | None = None
    translated_email: str | None = None
    order_id: str | None = None
    isorder_valid: bool | None = None
    email_category: str | None = None
    email_response: str | None = None
    final_status: str | None = None  # "actioned", "ignored"

    # Control flags
    hitl_required: bool = False
    validation_complete: bool = False

async def get_email_details_mcp(message_Id: str):
    """Get email details fromMCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        print(tools)
        
        # Find the getEmailByMessageId update tool
        getemaildetails_tool = next((tool for tool in tools if "getemailbymessageid" in tool.name.lower()), None)
        if not getemaildetails_tool:
            logging.error("Get Email by message id tool not found in MCP server")
            raise Exception("Get Email by message id tool not available")
        
        try:
            result = await getemaildetails_tool.ainvoke({
                "in_EmailMessageId": message_Id
            })
            print(result+"-----------------"+message_Id)
            logging.info(f"Retrieved email details via MCP")        
            # Assuming 'result' is your string variable
            email_details_dict = ast.literal_eval(result) if result else None  
            return email_details_dict if email_details_dict else None   
            
        except Exception as e:
            logging.error(f"Error retrieving email details via MCP: {e}")
            raise
# ---------------- Graph Nodes ----------------
class GraphOutput(BaseModel):
    report: str

async def get_email_details_node(state: GraphState) -> GraphOutput:
    """Get email details via MCP integration"""
    email_details = await get_email_details_mcp(
        state.message_id
    )
    
    return state.model_copy(update={
        "email_subject": email_details['email_subject'] or None,
        "email_body": email_details['email_body'] or None
    })

# ---------------- Nodes ----------------
async def start_node(state: GraphState) -> GraphState:
    """Extract order id information from the request"""
    system_prompt = """You are a data extraction expert tasked with extracting order id information from input text. 

    Your goal is to extract the following three fields:
    1. order id - the order id will be 8 digit number

    Instructions:
    - Only return a JSON object with keys: order_id
    - If a field cannot be determined, return null.
    - If multiple order ids are present, extract the first one only.
    - Only output the JSON. Do not include any explanations, commentary, or extra text.

    Examples:

    User message: "My order number is 12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }

    User message: "I have purchased a fan with order id 87654321 last week."
    Output:
    {
    "order_id": "87654321",
    }

    User message: "My TV isn't working properly."
    Output:
    {
    "order_id": null
    }
    """

    output = await llm.ainvoke(
        [SystemMessage(system_prompt),
         HumanMessage(state.email_body)]
    )

    order_id = json.loads(output.content)
    
    return state.model_copy(update={
        "order_id": order_id.get("order_id")
    })

def end_node(state: GraphState) -> GraphState:
    """Final node to log the completion"""
    logging.info(f"Email processing completed. Status: {state.order_id}")
    return state
# ---------------- Build Graph ----------------
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("step_start", start_node)
graph.add_node("get_email_details", get_email_details_node)
graph.add_node("step_end", end_node)

# Set entry point
graph.set_entry_point("step_start")
# Add edges
graph.add_edge("step_start", "get_email_details")
graph.add_edge("get_email_details", "step_end")
graph.add_edge("step_end", END)

# Compile the graph
agent = graph.compile()
