---
title:  Langchain/Mem0/ReAct Agent
date: 2026-03-07 21:00:00 +0800
categories: [Langchain, Langgraph, Mem0, Agent, Mcp]
tags: [agent, mcp, mem0, langchain, langgraph]
pin: true
---


## Agent ReAct Flow

LLM 永远不直接执行任何 side-effect，所有执行权归 Runtime 状态机。


```
                            User Input
                                │
                                ▼
                         ┌─────────────┐
                         │ Intent Gate │── chitchat ─▶ 直接回答
                         └──────┬──────┘── reject  ─▶ 拒绝
                                │ tool_needed
                                ▼
                         ┌─────────────┐
        ┌───────────────▶│    PLAN     │◀──────── replan ──────┐
        │                │   (调LLM)   │                       │
        │                └──────┬──────┘                       │
        │                       │ proposal                     │
        │                       ▼                              │
        │                ┌─────────────┐                       │
        │                │ DECIDE      │  steps?               │
        │                └──────┬──────┘                       │
        │                       │                              │
        │              ┌────────┴────────┐                     │
        │              ▼                 ▼                     │
        │        action="tool"     action="answer"             │
        │              │                 │                     │
        │              ▼                 ▼                     │
        │        ┌──────────┐     ┌──────────────┐            │
        │        │Validator │     │Answer        │            │
        │        │exists?   │     │Validator     │            │
        │        │args?     │     │(防幻觉检查)   │            │
        │        │          │     └──────┬───────┘            │
        │        │dup/cost? │       ┌────┴────┐               │
        │        └────┬─────┘       ▼         ▼               │
        │             │           PASS    FAIL ─── replan ───▶│
        │        ┌────┴────┐        │                         │
        │        ▼         ▼        │                         │
        │      PASS      FAIL       │                         │
        │        │    ┌────┴────┐    │                         │
        │        │    ▼         ▼    │                         │
        │        │   超限 FINAL      否    │                         │
        │        │       replan ────┼────────────────────────▶│
        │        │                 │                         │
        │        ▼                 ▼                         │
        │      ┌────────┐   ┌──────────────────┐             │
        │      │  ACT   │   │  Reflector 终审   │             │
        │      │(执行tool)│  │ (最终答案质量检查) │             │
        │      └───┬────┘   └────────┬─────────┘             │
        │          │            ┌────┴────┐                   │
        │          ▼            ▼         ▼                   │
        │      ┌──────────┐   PASS    FAIL ─── replan ───────▶│
        │      │ OBSERVE  │     │                             
        │      │(记录结果) │     │                             
        │      └───┬──────┘     │                             
        │          │            ▼                              
        └──────────┘       ┌──────────┐                        
       (回PLAN,下一步)      │  FINAL   │◀── 步数/预算超限        
                           │ (生成回答) │   (跳过终审)           
                           └────┬─────┘                        
                                │                              
                                ▼                              
                           User Answer                         
                                                               
```

```

Intent Gate
  快速短路，没必要走状态机的响应。
BERT小模型分类器 ~5-15ms  精度高成本低。 
Fine-tune 一个轻量模型做三分类 `{tool_needed, chitchat, reject}`

Context Builder


User Input
    │
    ▼
┌─────────────┐  命中 → 直接返回
│  规则匹配层  │  (reject / 明确 chitchat)
└──────┬──────┘
       │ 未命中
       ▼
┌─────────────┐  置信度 > 0.9 → 直接返回
│  小模型分类   │  (DistilBERT, 比 BERT 快 3x，精度只掉 3% ~10ms)
└──────┬──────┘
       │ 置信度 < 0.9（模糊区间）
       ▼
┌─────────────┐  兜底判断
│  LLM 轻判断  │  (GPT-4o-mini)
└─────────────┘

if intent == "chitchat":
    return chitchat_llm(user_input)
# guard_rails
if intent == "reject": 
    return "抱歉，无法处理该请求。"


Answer Validator
    Tool Necessity Check（工具必要性
    Confidence Check（置信度检查）
    避免幻觉 没有走工具。 

Runtime Validator 五重校验:
  1. tool_exists?      → 不存在 → PLAN (error feedback)
  2. args_valid?       → schema不匹配 → PLAN (error feedback)  
  3. step_limit?       → 超限 → FINAL (force answer)
  4. duplicate_call?   → 重复 → PLAN (skip + warn)
  5. cost_budget?      → 超预算 → FINAL (force answer)

Decide层做厚：
    `allowed = system ∩ tenant ∩ session`
    审计与合规（Audit Trail）save
    Guardrails
    FAIL  → 不执行，构造 error_feedback， PLAN，避免plan 问题

Executor
    parallel execution

Memory Layer

其它优化： 

Step Reflector 加一层更细的check 
 tool_result
     │
     ▼
reflector
     │
     ├─ accept
     └─ replan

Speculative Planning:
    LLM 一次 plan: [step1 查航班, step2 查日历, step3 创建事件]
    execute search_flight  │  execute get_calendar
        execute create_event
本质是 多步并行。 

Knowledge Graph
    实体、关系、事实的结构化表示

再加一层 Multi-Agent Orchestrator
     coordinate agent collaboration。


```

## Mcp tool example 

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

## Agent with rag and tools

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
# Langraph Mem0 Fastapi Template

## Langraph
```
Graph图：
如果跑图一半恢复，需要checkpointer,AsyncPostgresSaver是个例子，考虑性能 流量 最好换。 
self._graph = graph_builder.compile(checkpointer=checkpointer
```
## Mem0 长期记忆 
```
Mem0长期记忆：
async def _long_term_memory(self) -> AsyncMemory:
用 mem0 长期记忆 需要 AsyncMemory，需要提供 vector_store的provider. 
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    vector vector(1536),   -- 嵌入向量，维度由 embedder 决定 text-embedding-3-small 默认 1536
    payload JSONB          -- 见下
);

筛出行向量相似度检索。
跑完图(flow)执行 _update_long_term_memory，后台更新长期记忆。 

clear_chat_history 清除 ["checkpoint_blobs", "checkpoint_writes", "checkpoints"] 3个表中thread_id 内容,仅 checkpoint，不删 mem0。 
```
## 大致flow 
```
create_session 后 /chat 根据jwt token拿session_id 
 接口带 slowapi限流 
session_id = verify_token(token)

拿记忆，prepare_messages 会压缩， agent跑图，更新 长期记忆。 
chat_history来自 checkpointer。
```
