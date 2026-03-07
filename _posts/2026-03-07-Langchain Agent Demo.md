---
title:  Langchain Agent Demo
date: 2026-03-07 21:00:00 +0800
categories: [Langchain, Agent, Mcp]
tags: [agent, mcp, langchain]
pin: true
---

# Mcp tool example 

```python
mcp = FastMCP("mcp001")


@mcp.tool()
def get_weather(city: str) -> str:
    weather_db = {
        "北京": "晴天，25°C",
        "杭州": "晴天，23°C"
    }
    return weather_db.get(city, f"没有{city}的天气数据")


if __name__ == "__main__":
    mcp.run(transport='streamable-http', host='0.0.0.0', port=9000)
```

# Agent with rag and tools

```python
MCP_URL = "http://127.0.0.1:9000/mcp"
LLM_MODEL = "deepseek-ai/DeepSeek-V3.2"
EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
MAX_HISTORY_ROUNDS = 20

## llm and embeddings
def build_llm(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(  
        model=LLM_MODEL,
        base_url=BASE_URL,
        api_key=api_key,
        timeout=30,
        max_retries=2,
    )

def build_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=api_key,
    )

## rag
def build_retriever(embeddings: OpenAIEmbeddings) -> Any:
   # 可加持久化
    vectorstore = Chroma.from_documents(
        [Document(page_content=doc) for doc in KB_DOCS],
        embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})


## tools 
@tool
def multiply(a: float, b: float) -> float:
    """将两个数字相乘。"""
    return a * b

## mcp tools
async def get_mcp_tools() -> list[Any]:
    client = MultiServerMCPClient(
        {"weather": {"url": MCP_URL, "transport": "streamable_http"}}
    )
    try:
        async with client.session("weather") as session:
            return await load_mcp_tools(session, server_name="weather")
    except Exception as e:
        log.warning("MCP服务不可用，跳过天气工具: %s", e)
        return []


## sessions 
sessions: dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]

def trim_history(session_id: str):
    history = get_session_history(session_id)
    msgs = history.messages
    max_msgs = MAX_HISTORY_ROUNDS * 2
    if len(msgs) > max_msgs:
        history.clear()
        for m in msgs[-max_msgs:]:
            history.add_message(m)

## prompt 
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手,根据知识库检索和tools查询，来回答用户的问题。"),
    ("human", "以下是知识库检索的信息：\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

## agent

def build_agent_with_history(llm: ChatOpenAI, tools: list[Any]) -> RunnableWithMessageHistory:
    agent = create_tool_calling_agent(llm, tools, PROMPT)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        max_execution_time=60,
        handle_parsing_errors=True,
    )

    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

## conversations example

CONVERSATIONS = [
    "LangChain是什么？",
    "Chroma是什么？",
    "FastMCP是什么？",
    "帮我算一下1024乘以768等于多少？",
    "帮我查一下北京今天的天气怎么样？",
    "总结下我们刚刚的所有对话内容",
]

async def run_conversations(
    agent_with_history: RunnableWithMessageHistory,
    retriever: Any,
    session_id: str,
):
    config = {"configurable": {"session_id": session_id}}
    for q in CONVERSATIONS:
        docs = retriever.invoke(q)
        context = "\n".join(doc.page_content for doc in docs).strip()
        if not context:
            context = "无相关知识库结果"
        try:
            resp = await agent_with_history.ainvoke(
                {"input": q, "context": context},
                config=config,
            )
            print(f"\n{resp['output']}")
        except Exception as e:
            log.error("Agent 调用失败: %s", e)
            print(f"\n[错误] {e}")
        trim_history(session_id)

async def agent001():
    api_key = get_api_key()
    llm = build_llm(api_key)
    embeddings = build_embeddings(api_key)
    retriever = build_retriever(embeddings)
    mcp_tools = await get_mcp_tools()
    tools = [multiply] + mcp_tools
    agent_with_history = build_agent_with_history(llm, tools)
    await run_conversations(agent_with_history, retriever, SESSION_ID)

if __name__ == "__main__":
    asyncio.run(agent001())

```

