import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import json

# LangChain 导入
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Tavily 搜索
from tavily import TavilyClient

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

# ======================== 联网搜索 ========================

tavily_api_key = get_secret("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def search_web(query):
    """使用 Tavily 搜索互联网"""
    try:
        if tavily_client is None:
            return "未配置 TAVILY_API_KEY，无法执行联网搜索。"

        result = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=SEARCH_MAX_RESULTS
        )
        results = result.get("results", [])
        if not results:
            return "没有找到相关搜索结果。"

        formatted_results = []
        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            formatted_results.append(f"标题: {title}\n链接: {url}\n摘要: {content}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"搜索失败: {str(e)}"


# ======================== 页面配置 ========================

st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={}
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
    unsafe_allow_html=True
)

# ======================== API 初始化 ========================

api_key = get_secret("DEEPSEEK_API_KEY")
if not api_key:
    st.error("❌ DEEPSEEK_API_KEY 未配置，请在 Settings → Secrets 中添加")
    st.stop()

deepseek_base_url = get_secret("DEEPSEEK_BASE_URL") or get_secret("DEEPSEEK_API_BASE") or "https://api.deepseek.com"

# 初始化 LangChain ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base=deepseek_base_url,
    max_tokens=MAX_OUTPUT_TOKENS,
    streaming=False  # 先不流式，后面手动流式处理
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
        # 将 LangChain 消息对象转为可序列化格式
        serializable_messages = []
        for msg in st.session_state.messages:
            if isinstance(msg, SystemMessage):
                serializable_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                serializable_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                serializable_messages.append({"role": "assistant", "content": msg.content})
            else:
                serializable_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        session_data = {
            "name": st.session_state.name,
            "nature": st.session_state.nature,
            "session_time_name": st.session_state.session_time_name,
            "messages": serializable_messages
        }
        if not os.path.exists("session"):
            os.mkdir("session")
        with open(f"session/{st.session_state.session_time_name}.json", "w", encoding="utf-8") as f:
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
        if os.path.exists(f"session/{session_time_name}.json"):
            with open(f"session/{session_time_name}.json", "r", encoding="utf-8") as f:
                session_data = json.load(f)
                st.session_state.name = session_data["name"]
                st.session_state.nature = session_data["nature"]
                st.session_state.session_time_name = session_time_name
                # 恢复为 LangChain 消息对象
                restored_messages = []
                for msg in session_data["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        restored_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        restored_messages.append(AIMessage(content=content))
                    elif role == "system":
                        restored_messages.append(SystemMessage(content=content))
                    else:
                        restored_messages.append(HumanMessage(content=content))
                st.session_state.messages = restored_messages
    except Exception:
        st.error("加载会话信息失败!")


def delete_session(session_time_name):
    if PUBLIC_MODE:
        return
    try:
        if os.path.exists(f"session/{session_time_name}.json"):
            os.remove(f"session/{session_time_name}.json")
            if session_time_name == st.session_state.session_time_name:
                st.session_state.messages = []
                st.session_state.session_time_name = get_session_time_name()
    except Exception:
        st.error("删除会话信息失败!")


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
                if st.button(session, width="stretch", icon="📒", key=f"load_{session}",
                             type="primary" if session == st.session_state.session_time_name else "secondary"):
                    load_session_data(session)
                    st.rerun()
            with col2:
                if st.button("", icon="❌", key=f"delete_{session}"):
                    delete_session(session)
                    st.rerun()

    st.divider()

    st.subheader("伴侣信息")
    name = st.text_input("姓名", placeholder="请输入伴侣姓名", value=st.session_state.name)
    if name != st.session_state.name:
        st.session_state.name = name
    nature = st.text_area("性格", placeholder="请输入伴侣性格", value=st.session_state.nature)
    if nature != st.session_state.nature:
        st.session_state.nature = nature

    st.divider()
    st.caption("💡 联网搜索已就绪 · 需要实时信息时将自动联网查询")

# ======================== 渲染聊天记录 ========================

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, SystemMessage):
        pass  # 不显示系统消息
    else:
        # 兼容旧格式
        if message.get("role") == "user":
            st.chat_message("user").write(message.get("content", ""))
        elif message.get("role") == "assistant":
            st.chat_message("assistant").write(message.get("content", ""))

# ======================== 消息输入 ========================

prompt = st.chat_input("请输入你的问题：")
if prompt:
    if len(prompt) > MAX_PROMPT_LENGTH:
        st.warning(f"这次输入太长了，请控制在 {MAX_PROMPT_LENGTH} 个字符以内。")
        st.stop()

    # 显示用户消息
    st.chat_message("user").write(prompt)
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)

    # ===================== 联网搜索 =====================
    web_context = ""
    if tavily_client is not None:
        web_context = search_web(prompt)

    # ===================== 构建系统提示 =====================
    system_prompt = f"""你叫{st.session_state.name}，现在是用户的真实伴侣，请完全代入伴侣角色。
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
    8.当用户询问今天日期、当前时间、星期几、今年是哪一年等本地时间问题时，必须直接根据当前本地日期时间回答
    9.当用户询问实时新闻、天气、最新事件、联网资料，或需要外部信息确认的问题时，优先使用提供的上下文回答
伴侣性格:
    10.回答必须正确，不能是错误答案
    {st.session_state.nature}
你必须严格遵守上述规则来回复用户。"""

    # 如果有联网搜索结果，加入系统提示
    if web_context:
        system_prompt += f"\n\n以下是联网搜索到的相关信息，请参考这些信息来回答：\n{web_context}"

    # ===================== 构建 LangChain 消息列表 =====================
    # 使用 LangChain 的 SystemMessage / HumanMessage / AIMessage 管理对话
    api_messages = [SystemMessage(content=system_prompt)]
    # 添加历史消息（最多 MAX_HISTORY_MESSAGES 条）
    history_messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
    for msg in history_messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            api_messages.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                api_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                api_messages.append(AIMessage(content=content))

    # ===================== 流式调用 LangChain ChatOpenAI =====================
    # 创建流式 LLM
    streaming_llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base=deepseek_base_url,
        max_tokens=MAX_OUTPUT_TOKENS,
        streaming=True,
        temperature=0.7
    )

    full_session_state = st.empty()
    full_response = ""

    try:
        # 使用 LangChain 的流式接口
        stream = streaming_llm.stream(api_messages)
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
    st.session_state.messages.append(AIMessage(content=full_response))
    save_session()