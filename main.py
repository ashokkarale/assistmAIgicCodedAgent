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
    agent_language: str | None = "English"
    email_body: str | None = None
    email_subject: str | None = None
    translated_email_body: str | None = None
    translated_email_subject: str | None = None
    mail_communication_language: str | None = None
    order_id: str | None = None
    order_details: Dict[str, Any] | None = None
    isorder_valid: bool | None = None
    email_category: str | None = None
    email_response: str | None = None
    final_status: str | None = None  # "actioned", "ignored"

    # Control flags
    hitl_required: bool = False
    validation_complete: bool = False

async def get_email_details_mcp(message_Id: str):
    """Get email details from MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        #print(tools)
        
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

##Translate email body and subject language if required
async def translate_email_language_mcp(email_subject: str, email_body: str, agent_language: str):
    """Translate email body and subject language if required by MCP tools"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        #print(tools)
        
        # Translate Email details tool
        translategetemaildetails_tool = next((tool for tool in tools if "translateemailsubjectandbodylanguage" in tool.name.lower()), None)
        if not translategetemaildetails_tool:
            logging.error("Translate Email Subject And BodybLanguage tool not found in MCP server")
            raise Exception("translateEmailSubjectAndBodyLanguage tool not available")
        
        print("Invoking translate tool...")
        print(agent_language+"-----------------"+email_subject+"-----------------"+email_body)
        try:
            result = await translategetemaildetails_tool.ainvoke({
                "in_OriginalEmailSubject": email_subject,
                "in_OriginalEmailBody": email_body,
                "in_Language": agent_language
            })
            print("translateemailsubjectandbodylanguage mcp tool result: "+"-----------------"+result)
            logging.info(f"Email language translation done via MCP")        
            # Assuming 'result' is your string variable
            email_details_dict = ast.literal_eval(result) if result else None  
            return email_details_dict if email_details_dict else None   
            
        except Exception as e:
            logging.error(f"Error translating email details via MCP: {e}")
            raise

##Get Order Details by order_id via MCP
async def get_order_details_mcp(order_id: str):
    """Get Order Details by order_id with MCP tool"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        #print(tools)
        
        # Get Order Details tool
        translategetemaildetails_tool = next((tool for tool in tools if "getOrderDetailsByOrderId".lower() in tool.name.lower()), None)
        if not translategetemaildetails_tool:
            logging.error("getOrderDetailsByOrderId tool not found in MCP server")
            raise Exception("getOrderDetailsByOrderId tool not available")
        
        print("Invoking get order details tool...")
        print(order_id+"-----------------")
        try:
            result = await translategetemaildetails_tool.ainvoke({
                "in_OrderNumber": order_id
            })
            print("getOrderDetailsByOrderId mcp tool result: "+"-----------------"+result)
            logging.info(f"Fetched order details via MCP tool.")        
            # Assuming 'result' is your string variable
            order_details_dict = ast.literal_eval(result) if result else None  
            return order_details_dict if order_details_dict else None   
            
        except Exception as e:
            logging.error(f"Error Fetching order details via MCP: {e}")
            raise


# ---------------- Graph Nodes ----------------
class GraphOutput(BaseModel):
    report: str
# ---------------- Nodes ----------------
# Get email details via MCP integration
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
# Translate email body and subject language if required via MCP integration
async def translate_email_language_node(state: GraphState) -> GraphOutput:
    """Translate email body and subject language if required by MCP integration"""
    email_details = await translate_email_language_mcp(
        state.email_subject,
        state.email_body,
        state.agent_language
    )
    
    return state.model_copy(update={
        "agent_language": email_details['out_CommunicationLanguage'] or None,
        "translated_email_body": email_details['out_TranslatedMailBody'] or None,
        "translated_email_subject": email_details['out_TranslatedMailSubject'] or None
    })

# ---------------- Nodes ----------------
# Extract order id from email body & subject   
async def extract_order_id_node(state: GraphState) -> GraphState:
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
    User message: "My order number is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order no is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order #12345678 and I need help."
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
    try:
        payload = json.loads(output.content)
    except Exception:
        payload = None

    raw = (output.content or "").lower()
    null_patterns = ['"order_id": null', "'order_id' is null", "order_id is null", "order_id: null"]

    if any(p in raw for p in null_patterns):
        
        extracted_order_id = None
    else:
        extracted_order_id = None
        if isinstance(payload, dict):
            extracted_order_id = payload.get("order_id")

    return state.model_copy(update={
        "order_id": extracted_order_id or None
    })

# ---------------- Nodes ----------------
# Get Order Details by order_id via MCP integration
async def get_order_details_node(state: GraphState) -> GraphOutput:
    """Get Order Details by order_id with MCP tool"""
    order_details_obj = await get_order_details_mcp(
        state.order_id
    )
    
    return state.model_copy(update={
        "order_details": order_details_obj["out_OrderDetails"] or None
    })

async def check_fields_node(state: GraphState) -> GraphState:
    """Check if all required fields are present"""
    hitl_required = not state.leave_start or not state.leave_end or not state.leave_reason
    return state.model_copy(update={"hitl_required": hitl_required})

def end_node(state: GraphState) -> GraphState:
    """Final node to log the completion"""
    logging.info(f"Email processing completed. Status: {state.final_status}")
    return state

# ---------------- Nodes ----------------
# Auto-reject email via MCP integration
async def auto_reject_node(state: GraphState) -> GraphState:
    """Send auto-rejection email with reason of rejectstion via MCP integration"""
    await reply_email_mcp(
        message_id=state.message_id,
        llmprompt_to_prepare_reply="We regret to inform you that your order Id is missing from your email. Please provide a valid order Id for us to assist you further.",
        reply_language=state.agent_language
    )
    
    return state.model_copy(update={"final_status": "completed"})

# ---------------- MCP Email Reply Function ----------------
async def reply_email_mcp(message_id: str, llmprompt_to_prepare_reply: str, reply_language: str):
    """Send email reply via MCP integration"""
    async with get_mcp_session() as session:
        tools = await load_mcp_tools(session)
        
        # Find the email tool
        email_tool = next((tool for tool in tools if "replyToEmail".lower() in tool.name.lower()), None)
        if not email_tool:
            logging.error("Email reply tool not found in MCP server.")
            raise Exception("Email reply tool not available.")
        
        try:
            await email_tool.ainvoke({
                "in_Message_Id": message_id,
                "in_llmprompt_to_prepare_reply": llmprompt_to_prepare_reply,
                "in_Reply_Language": reply_language
            })
            logging.info(f"Replied to email via MCP.")
            
        except Exception as e:
            logging.error(f"Error replying to email via MCP: {e}")
            raise

# ---------------- Nodes ----------------
# Categorize email via MCP integration
async def categorize_email_node(state: GraphState) -> GraphOutput:
    """Categorize email body and subject"""
    system_prompt = """You are a after sales product and service support expert tasked with extracting order id information from input text. 

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
    User message: "My order number is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order no is #12345678 and I need help."
    Output:
    {
    "order_id": "12345678"
    }
    User message: "My order #12345678 and I need help."
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
    try:
        payload = json.loads(output.content)
    except Exception:
        payload = None

    raw = (output.content or "").lower()
    null_patterns = ['"order_id": null', "'order_id' is null", "order_id is null", "order_id: null"]

    if any(p in raw for p in null_patterns):
        
        extracted_order_id = None
    else:
        extracted_order_id = None
        if isinstance(payload, dict):
            extracted_order_id = payload.get("order_id")

    return state.model_copy(update={
        "order_id": extracted_order_id or None
    })


# ---------------- Condition Functions ----------------
def should_go_to_order_id_auto_reject(state: GraphState):
    """Check if order_id is missing and autorejection is required"""
    return "order_id_missing" if state.order_id is None else "order_id_available"

# ---------------- Build Graph ----------------
graph = StateGraph(GraphState)

# Add all nodes
graph.add_node("translate_email_language", translate_email_language_node) #first node
graph.add_node("extract_order_id", extract_order_id_node)
graph.add_node("get_order_details", get_order_details_node)
graph.add_node("auto_reject", auto_reject_node)
graph.add_node("categorize_email", categorize_email_node)
graph.add_node("step_end", end_node)

# Set entry point
graph.set_entry_point("translate_email_language")
# Add edges
graph.add_edge("translate_email_language", "extract_order_id")
#graph.add_edge("extract_order_id", "get_order_details")
graph.add_conditional_edges(
    "extract_order_id", 
    should_go_to_order_id_auto_reject,
    {
        "order_id_missing": "auto_reject",
        "order_id_available": "get_order_details"
    }
 )
graph.add_edge("auto_reject", "step_end")
graph.add_edge("get_order_details", "categorize_email")
graph.add_edge("categorize_email", "step_end")
graph.add_edge("step_end", END)

# Compile the graph
agent = graph.compile()
