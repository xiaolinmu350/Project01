"""
AI 智能伴侣 — 基于 LangChain + LangGraph 的 AI 智能体
=====================================================
使用 LangChain 的 ChatOpenAI 调 DeepSeek API
使用 LangGraph (StateGraph) 构建智能体工作流
Tavily 搜索作为智能体的 Tool
Streamlit 做前端界面
"""

import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import TypedDict, Annotated, Sequence

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ======================== 配置和初始化 ========================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, ".env"))


def get_secret(name, default=None):
    try:
        val = st.secrets.get(name, os.environ.get(name, None))
        return val if val else os.environ.get(name, default)
    except Exception:
        return os.environ.get(name, default)


def is_enabled(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


PUBLIC_MODE = is_enabled(get_secret("PUBLIC_MODE", "false"))
MAX_PROMPT_LENGTH = 1000 if PUBLIC_MODE else 4000
MAX_HISTORY_MESSAGES = 12 if PUBLIC_MODE else 30
MAX_OUTPUT_TOKENS = 500 if PUBLIC_MODE else 1000
SEARCH_MAX_RESULTS = 3 if PUBLIC_MODE else 5

# API Keys
deepseek_api_key = get_secret("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    st.error("❌ DEEPSEEK_API_KEY 未配置，请在 Settings → Secrets 中添加")
    st.stop()

deepseek_base_url = (
    get_secret("DEEPSEEK_BASE_URL")
    or get_secret("DEEPSEEK_API_BASE")
    or "https://api.deepseek.com"
)

tavily_api_key = get_secret("TAVILY_API_KEY")

# ======================== LangChain 工具定义 ========================

# 使用 langchain_tavily 的官方封装（LangChain Tool）
tavily_search_instance = (
    TavilySearch(
        tavily_api_key=tavily_api_key,
        max_results=SEARCH_MAX_RESULTS,
    )
    if tavily_api_key
    else None
)


@tool
def search_web(query: str) -> str:
    """搜索实时信息、新闻、天气、知识类问题或需要联网确认的内容。"""
    if tavily_search_instance is None:
        return "未配置 TAVILY_API_KEY，无法执行联网搜索。"
    try:
        results = tavily_search_instance.invoke({"query": query})
        if not results:
            return "没有找到相关搜索结果。"
        # TavilySearch 返回的是 list[dict] 或 list[str]
        if isinstance(results, list):
            formatted = []
            for r in results:
                if isinstance(r, dict):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    content = r.get("content", "")
                    formatted.append(f"标题: {title}\n链接: {url}\n摘要: {content}")
                elif isinstance(r, str):
                    formatted.append(r)
            return "\n\n".join(formatted) if formatted else "没有找到相关搜索结果。"
        return str(results)
    except Exception as e:
        return f"搜索失败: {str(e)}"


tools = [search_web]

# ======================== LangGraph 智能体构建 ========================


class AgentState(TypedDict):
    """LangGraph 状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url=deepseek_base_url,
    temperature=0.7,
    max_tokens=MAX_OUTPUT_TOKENS,
    streaming=False,  # 图内调用不需要 streaming
)

# 绑定工具到 LLM（LangChain 的 bind_tools → 自动生成 Function Calling 格式）
llm_with_tools = llm.bind_tools(tools)
# ======================== 构建智能体 ========================

agent_graph = create_react_agent(llm, tools)

# ======================== 页面配置 ========================

st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={},
)

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .stChatInput textarea {
            font-size: 16px !important;
        }
        .stButton button {
            min-height: 44px !important;
        }
        [data-testid="stChatMessage"],
        [data-testid="stChatMessage"] p {
            font-size: 16px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================== 辅助函数 ========================


def get_session_time_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_current_datetime_context():
    now = datetime.now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S，{weekday_names[now.weekday()]}")


def save_session():
    if PUBLIC_MODE:
        return
    if st.session_state.session_time_name:
        session_data = {
            "name": st.session_state.name,
            "nature": st.session_state.nature,
            "session_time_name": st.session_state.session_time_name,
            "messages": st.session_state.messages,
        }
        if not os.path.exists("session"):
            os.mkdir("session")
        with open(
            f"session/{st.session_state.session_time_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)


def load_session():
    if PUBLIC_MODE:
        return []
    session_list = []
    if os.path.exists("session"):
        file_list = os.listdir("session")
        for file in file_list:
            if file.endswith(".json"):
                session_list.append(file[:-5])
    session_list.sort(reverse=True)
    return session_list


def load_session_data(session_time_name):
    if PUBLIC_MODE:
        return
    try:
        filepath = f"session/{session_time_name}.json"
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                session_data = json.load(f)
                st.session_state.name = session_data["name"]
                st.session_state.nature = session_data["nature"]
                st.session_state.session_time_name = session_time_name
                st.session_state.messages = session_data["messages"]
    except Exception:
        st.error("加载会话信息失败!")


def delete_session(session_time_name):
    if PUBLIC_MODE:
        return
    try:
        filepath = f"session/{session_time_name}.json"
        if os.path.exists(filepath):
            os.remove(filepath)
            if session_time_name == st.session_state.session_time_name:
                st.session_state.messages = []
                st.session_state.session_time_name = get_session_time_name()
    except Exception:
        st.error("删除会话信息失败!")


# ======================== LangGraph 智能体调用 ========================


def run_agent(messages: list) -> str:
    """
    运行 LangGraph 智能体，返回最终回答。

    """
    # 将 dict 消息列表转为 LangChain Message 对象
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    # 运行 LangGraph 图
    result = agent_graph.invoke({"messages": lc_messages})

    # 从结果中提取最终 AIMessage 内容
    final_messages = result["messages"]
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "抱歉，我没有找到合适的回答。"


# ======================== 页面标题 ========================

st.title("AI智能伴侣")
st.logo("👾")

# ======================== 会话状态初始化 ========================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name" not in st.session_state:
    st.session_state.name = "星婉"

if "nature" not in st.session_state:
    st.session_state.nature = "温柔体贴的妹妹"

if "session_time_name" not in st.session_state:
    st.session_state.session_time_name = get_session_time_name()

# ======================== 侧边栏 ========================

with st.sidebar:
    st.subheader("AI控制面板")
    if st.button("新建会话", width="stretch", icon="✏️"):
        save_session()
        if st.session_state.messages:
            st.session_state.messages = []
            st.session_state.session_time_name = get_session_time_name()
            save_session()
            st.rerun()

    if PUBLIC_MODE:
        st.caption("公开模式不会保存或展示共享历史会话。")
    else:
        st.text("历史会话")
        session_list = load_session()
        for session in session_list:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    session,
                    width="stretch",
                    icon="📒",
                    key=f"load_{session}",
                    type="primary"
                    if session == st.session_state.session_time_name
                    else "secondary",
                ):
                    load_session_data(session)
                    st.rerun()
            with col2:
                if st.button("", icon="❌", key=f"delete_{session}"):
                    delete_session(session)
                    st.rerun()

    st.divider()

    st.subheader("伴侣信息")
    name = st.text_input(
        "姓名", placeholder="请输入伴侣姓名", value=st.session_state.name
    )
    if name != st.session_state.name:
        st.session_state.name = name
    nature = st.text_area(
        "性格", placeholder="请输入伴侣性格", value=st.session_state.nature
    )
    if nature != st.session_state.nature:
        st.session_state.nature = nature

    st.divider()
    st.caption("💡 LangGraph 智能体引擎 · 自动联网查询")

# ======================== 渲染聊天记录 ========================

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])

# ======================== 消息输入 ========================

prompt = st.chat_input("请输入你的问题：")
if prompt:
    if len(prompt) > MAX_PROMPT_LENGTH:
        st.warning(f"这次输入太长了，请控制在 {MAX_PROMPT_LENGTH} 个字符以内。")
        st.stop()

    # 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 构建系统提示
    system_content = f"""你叫{st.session_state.name}，现在是用户的真实伴侣，请完全代入伴侣角色。
当前本地日期时间:
    {get_current_datetime_context()}
规则:
    1.每次只回1条消息
    2.禁止任何场景或状态描述性文字
    3.匹配用户的语言
    4.回复简短，像微信聊天一样
    5.有需要的话可以用等emoji表情
    6.用符合伴侣性格的方式对话
    7.回复的内容，要充分体现伴侣的性格特征
    8.当用户询问今天日期、当前时间、星期几、今年是哪一年等本地时间问题时，必须直接根据当前本地日期时间回答，不使用 search_web 工具
    9.当用户询问实时新闻、天气、最新事件、联网资料，或需要外部信息确认的问题时，使用 search_web 工具搜索后再回答
伴侣性格:
    10.回答必须正确，不能是错误答案
    {st.session_state.nature}
你必须严格遵守上述规则来回复用户。"""

    # 构建消息列表
    api_messages = [{"role": "system", "content": system_content}]
    history_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    for msg in history_messages:
        api_messages.append(msg)

    # ===================== LangGraph 智能体调用 =====================
    try:
        with st.spinner("思考中..."):
            # 第1步：运行 LangGraph 智能体，获取推理结果
            final_answer = run_agent(api_messages)

        # 第2步：流式输出最终回答（使用 LangChain 的流式接口）
        full_session_state = st.empty()
        full_response = ""

        stream = llm.stream(
            [
                SystemMessage(
                    content=system_content
                ),
                HumanMessage(
                    content=f"请根据以下信息回答用户的问题（用伴侣的性格和语气，简短自然）：\n\n{final_answer}"
                ),
            ]
        )
        for chunk in stream:
            if chunk.content:
                content = chunk.content
                full_response += content
                full_session_state.chat_message("assistant").write(full_response)

        if not full_response:
            full_response = "抱歉，我现在无法回答，请稍后再试。"
            full_session_state.chat_message("assistant").write(full_response)

    except Exception as e:
        full_response = f"请求出错: {str(e)}"
        st.error(full_response)

    # 保存回复到会话
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_session()