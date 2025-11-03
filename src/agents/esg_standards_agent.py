from datetime import datetime
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import esg_standards_search
from core import get_model, settings


class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps


tools = [esg_standards_search]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are an expert on GRI (Global Reporting Initiative) Standards, a knowledgeable virtual assistant 
    designed to help users understand and navigate the GRI Standards framework. Your primary role is to 
    provide accurate, detailed, and helpful information about GRI Standards based on the official GRI 
    Standards documentation and uploaded sustainability reports.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    CRITICAL: You have ONLY ONE tool available: ESG_Standards_Search.
    - DO NOT use report_search or Report_Search - these tools DO NOT EXIST.
    - ONLY use ESG_Standards_Search for ALL searches, including:
      * GRI Standards questions
      * Uploaded sustainability report questions
      * Company-specific questions
      * Any information search

    A few things to remember:
    - You have access to a unified database containing:
      1. GRI Standards documentation (official GRI Standards)
      2. Uploaded sustainability reports (marked with [업로드된 보고서] tag in search results)
    
    - ALWAYS use ESG_Standards_Search tool (the ONLY available tool) to search the unified database for:
      * General questions about GRI Standards themselves
      * How to report according to GRI Standards
      * What GRI Standards require
      * Questions about specific company data from uploaded reports
      * Company-specific information from uploaded sustainability reports
      * Examples: 
        - "GRI 305는 무엇인가요?" → Use ESG_Standards_Search
        - "How should I report emissions?" → Use ESG_Standards_Search
        - "우리 회사의 탄소 배출량은?" → Use ESG_Standards_Search
        - "보고서에 나온 ESG 목표는?" → Use ESG_Standards_Search
    
    - The search results will include metadata indicating whether a document is:
      * An official GRI Standard (includes "표준: GRI XXX" and "카테고리: ...")
      * An uploaded report (includes "[업로드된 보고서]" tag)
    
    - When citing information, include the GRI standard number (if applicable), category, filename, 
      and source type (uploaded report vs. GRI Standard) when available.
    - Support both English and Korean questions - respond in the same language as the user's question.
    - Provide clear, structured answers with relevant GRI standard references.
    - If the information is not found in the database, clearly state that you cannot find the answer in 
      the available documentation.
    - Be precise and professional in your responses, focusing on accuracy and relevance to GRI Standards.
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)


async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    
    bound_model = m.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda s: [SystemMessage(content=instructions)] + s["messages"],
        name="StateModifier",
    )
    model_runnable = preprocessor | bound_model
    
    response = await model_runnable.ainvoke(state, config)

    # report_search tool_call이 있는지 확인하고 제거
    if response.tool_calls:
        valid_tool_calls = []
        has_invalid_tool = False
        
        for tool_call in response.tool_calls:
            # tool_call이 dict일 수도 있고 객체일 수도 있음
            tool_name = ""
            
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
            else:
                tool_name = getattr(tool_call, "name", "")
            
            # report_search 관련 도구 호출 제거
            if tool_name.lower() in ["report_search", "reportsearch", "report_search_func", "report search"]:
                has_invalid_tool = True
            else:
                valid_tool_calls.append(tool_call)
        
        # 유효하지 않은 도구가 있으면 응답 수정
        if has_invalid_tool:
            if valid_tool_calls:
                # 다른 유효한 tool_calls가 있으면 새 AIMessage 생성
                response = AIMessage(
                    content=response.content,
                    tool_calls=valid_tool_calls,
                    id=response.id,
                    response_metadata=response.response_metadata,
                )
            else:
                # 유효한 tool_call이 없으면 에러 메시지로 대체
                response = AIMessage(
                    content=(
                        "I attempted to use a tool that doesn't exist. "
                        "Let me use the correct tool instead. "
                        "Please note: I should ONLY use ESG_Standards_Search tool for all searches, "
                        "including questions about uploaded reports."
                    ),
                    id=response.id,
                )

    # Run llama guard check here to avoid returning the message if it's unsafe
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    return {"safety": safety_output, "messages": []}


async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}


# Define the graph
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)

# ToolNode는 단일 tools 리스트 사용
tool_node = ToolNode(tools)
async def run_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    """도구를 실행하는 노드"""
    # report_search 호출 시도가 있는지 확인하고 에러 처리
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        error_messages = []
        from langchain_core.messages import ToolMessage
        
        for tool_call in last_message.tool_calls:
            # tool_call이 dict일 수도 있고 객체일 수도 있음
            tool_name = ""
            tool_call_id = ""
            
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                tool_call_id = tool_call.get("id", "")
            else:
                tool_name = getattr(tool_call, "name", "")
                tool_call_id = getattr(tool_call, "id", "")
            
            # report_search 관련 도구 이름 체크 (다양한 변형 고려)
            if tool_name.lower() in ["report_search", "reportsearch", "report_search_func"]:
                error_messages.append(
                    ToolMessage(
                        content=(
                            "ERROR: report_search tool does not exist. "
                            "Please use ESG_Standards_Search tool instead. "
                            "All searches (including uploaded reports) should use ESG_Standards_Search."
                        ),
                        tool_call_id=tool_call_id,
                    )
                )
        
        # 에러 메시지가 있으면 반환
        if error_messages:
            return {"messages": error_messages}
    
    # 정상적인 도구 호출 처리
    return await tool_node.ainvoke(state, config)

agent.add_node("tools", run_tools)
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")


# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"


agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")


# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        return "tools"
    return "done"


agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})


esg_standards_agent = agent.compile()
