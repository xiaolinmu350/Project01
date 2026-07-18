import streamlit as st
import os
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime
import json

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


# 页面配置项
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
# 会话标识函数
def get_session_time_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_current_datetime_context():
    now = datetime.now()
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S，{weekday_names[now.weekday()]}")
# 保存会话信息函数
def save_session():
    if PUBLIC_MODE:
        return
    # 1.保存当前会话信息
    if st.session_state.session_time_name:
        # 构建新的会话对象
        session_data = {
            "name": st.session_state.name,
            "nature": st.session_state.nature,
            "session_time_name": st.session_state.session_time_name,
            "messages": st.session_state.messages
        }
        # 如果session文件夹不存在，则创建
        if not os.path.exists("session"):
            os.mkdir("session")
        # 保存会话数据
        with open(f"session/{st.session_state.session_time_name}.json", "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
# 页面展示加载会话信息函数
def load_session():
    if PUBLIC_MODE:
        return []
    session_list = []
    if os.path.exists("session"):
        file_list = os.listdir("session")
        for file in file_list :
            if file.endswith(".json"):
                session_list.append(file[:-5])
    session_list.sort(reverse=True)    # 降序排列
    return session_list
# 加载当前指定会话信息函数
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
                st.session_state.messages = session_data["messages"]
    except Exception:
        st.error("加载会话信息失败!")
# 删除指定会话函数
def delete_session(session_time_name):
    if PUBLIC_MODE:
        return
    try:
        if os.path.exists(f"session/{session_time_name}.json"):
            os.remove(f"session/{session_time_name}.json") # 删除文件
            # 如果删除的是当前会话，页面重新加载
            if session_time_name == st.session_state.session_time_name:
                st.session_state.messages = []
                st.session_state.session_time_name = get_session_time_name()

    except Exception:
        st.error("删除会话信息失败!")


def search_web(query):
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


search_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索实时信息、新闻、天气、知识类问题或需要联网确认的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的关键词或问题"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# 大标题
st.title("AI智能伴侣")
st.logo("👾")
# 创建OpenAI客户端
api_key = get_secret("DEEPSEEK_API_KEY")
if not api_key:
    st.error("❌ DEEPSEEK_API_KEY 未配置，请在 Settings → Secrets 中添加")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(
    api_key=api_key,
    base_url=get_secret("DEEPSEEK_BASE_URL") or get_secret("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
)
tavily_api_key = get_secret("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None
# 系统提示词
system_prompt = """
                你叫%s，现在是用户的真实伴侣，请完全代入伴侣角色。
                当前本地日期时间:
                    %s
                规则:
                    1.每次只回1条消息
                    2.禁止任何场景或状态描述性文字
                    3.匹配用户的语言
                    4.回复简短，像微信聊天一样
                    5.有需要的话可以用等emoji表情
                    6. 用符合伴侣性格的方式对话
                    7.回复的内容,要充分体现伴侣的性格特征
                    8.当用户询问今天日期、当前时间、星期几、今年是哪一年等本地时间问题时，必须直接根据当前本地日期时间回答，不使用 search_web
                    9.当用户询问实时新闻、天气、最新事件、联网资料，或需要外部信息确认的问题时，优先使用 search_web 工具搜索后再回答
                伴侣性格:
                    10.回答必须正确，不能是错误答案
                    %s
                    你必须严格遵守上述规则来回复用户。"""

# 初始化聊天信息
if "messages" not in st.session_state:
    st.session_state.messages = []
# 初始化名称
if "name" not in st.session_state:
    st.session_state.name = "小明"

# 初始化性格
if "nature" not in st.session_state:
    st.session_state.nature = "幽默的河南帅哥"

# 会话标识
if "session_time_name" not in st.session_state:
    st.session_state.session_time_name = get_session_time_name() # 当前时间

# 侧边栏
with st.sidebar:
    st.subheader("AI控制面板")
    if st.button("新建会话", width="stretch", icon="✏️"):
       # 1.保存当前会话信息
        save_session()

       # 2.创建会话信息
        if st.session_state.messages: # 如果聊天信息非空，为True，否则为False
            st.session_state.messages = []
            st.session_state.session_time_name = get_session_time_name()
            save_session()
            st.rerun () # 重新运行当前页面
    if PUBLIC_MODE:
        st.caption("公开模式不会保存或展示共享历史会话。")
    else:
        # 展示历史会话
        st.text("历史会话")
        session_list = load_session()
        for session in session_list:
            col1, col2 = st.columns([4,1])
            with col1:
                # 三元运算符：如果条件为真，则返回第一个值，否则返回第二个值 ---> 语法：值1 if 条件 else 值2
                if st.button (session,width="stretch", icon="📒",key=f"load_{session}",type="primary" if session == st.session_state.session_time_name else "secondary"):
                    load_session_data(session)
                    st.rerun()
            with col2:
                if st.button("",icon="❌", key=f"delete_{session}"):
                    delete_session(session)
                    st.rerun()

    # 分割线
    st.divider()


    st.subheader("伴侣信息")
    name = st.text_input("姓名",placeholder="请输入伴侣姓名",value=st.session_state.name)
    if name != st.session_state.name:
        st.session_state.name = name
    nature = st.text_area("性格",placeholder="请输入伴侣性格",value=st.session_state.nature)
    if nature != st.session_state.nature:
        st.session_state.nature = nature


#遍历聊天记录
for message in st.session_state.messages: #{"role": "user", "content": prompt}
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# 消息框
prompt = st.chat_input("请输入你的问题：")
if prompt: # 字符串自动转化为布尔值，输入内容不为空，则返回True
    if len(prompt) > MAX_PROMPT_LENGTH:
        st.warning(f"这次输入太长了，请控制在 {MAX_PROMPT_LENGTH} 个字符以内。")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    api_messages = [                                                       # 系统提示词
        {"role": "system", "content":system_prompt % (st.session_state.name,get_current_datetime_context(),st.session_state.nature)},
        *st.session_state.messages[-MAX_HISTORY_MESSAGES:] # 解包messages中存储的历史对话
    ]

    first_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        tools=search_tools,
        tool_choice="auto",
        max_tokens=MAX_OUTPUT_TOKENS
    )
    first_message = first_response.choices[0].message
    tool_calls = first_message.tool_calls or []

    if first_response.choices[0].finish_reason == "tool_calls" or tool_calls:
        api_messages.append(first_message.model_dump(exclude_none=True))
        for tool_call in tool_calls[:1]:
            if tool_call.function.name == "search_web":
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                query = arguments.get("query") or prompt
                search_result = search_web(query)
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": search_result
                })

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        stream=True,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    full_session_state = st.empty() # 创建一个空的占位符
    full_response = "" # 创建一个空的字符串
    for chunk in response:  # 遍历response
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            full_session_state.chat_message("assistant").write(full_response) # 更新占位符


    st.session_state.messages.append({"role": "assistant", "content":full_response})
    save_session()

