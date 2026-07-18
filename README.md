# 🤖 AI Partner

基于 LangChain + Streamlit 的 AI 智能伴侣对话系统，支持联网搜索、多轮对话、会话管理，部署在公网可访问。

## 功能亮点

- 🤖 **多轮对话** — 基于 DeepSeek + LangChain，支持上下文记忆
- 🔍 **联网搜索** — 集成 Tavily 搜索，自动获取实时信息
- 📱 **手机端适配** — 响应式界面，手机浏览器直接使用
- 💾 **会话管理** — 保存/加载/删除历史会话，支持切换伴侣角色
- 🎭 **自定义伴侣** — 可自由设置伴侣姓名和性格特征
- ⚡ **流式输出** — 实时显示 AI 回复，体验流畅

## 在线体验

👉 [点此访问](https://your-app-link.streamlit.app)

## 技术栈

| 技术 | 用途 |
|------|------|
| Python | 核心开发语言 |
| LangChain | LLM 调用与消息管理 |
| Streamlit | 前端界面框架 |
| DeepSeek-chat | 对话模型 |
| Tavily API | 联网搜索工具 |

## 本地运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
创建 `.env` 文件：
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
TAVILY_API_KEY=your_tavily_api_key
```

### 3. 运行
```bash
streamlit run "AI Partner.py"
```

## 项目结构
```
├── AI Partner.py        # 主程序
├── session/             # 会话数据存储
├── .env                 # 环境变量配置
├── .gitignore           # Git 忽略规则
├── requirements.txt     # 依赖清单
└── README.md            # 项目说明
```

## 部署到 Streamlit Cloud
1. 推送代码到 GitHub
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)
3. 部署入口设为 `AI Partner.py`
4. 在 Secrets 中配置 `DEEPSEEK_API_KEY`、`TAVILY_API_KEY` 等

## 许可证
MIT License