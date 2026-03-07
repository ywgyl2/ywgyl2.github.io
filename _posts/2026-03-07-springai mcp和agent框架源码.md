---
title:  SpringAi Mcp和Agent框架源码
date: 2026-03-07 22:00:00 +0800
categories: [SpringAi, Agent, Mcp]
tags: [agent, mcp, springai, langchain4j, java]
pin: true
---

# Backgroud 
整理下2025年中翻看的Spring ai源码，包括Agent框架 和 Mcp实现。也简单翻了下 Langchain4j。 


# Mcp Server & Client 

## tools 自动注册

注解 → JSON Schema
 本质 最终都是 适配到 io.modelcontextprotocol.sdk
```
`@Tool` + `@ToolParam`

│  MethodToolCallbackProvider.getToolCallbacks()                              │
│      ↓ ReflectionUtils.getDeclaredMethods → filter @Tool   
        MethodToolCallback
McpToolUtils.toSyncToolSpecification()  
    ToolCallback → SyncToolSpecification

```
 例子： 
```
MethodToolCallback{toolDefinition=DefaultToolDefinition[name=add, description=Add two integers together, inputSchema={
"$schema" : "https://json-schema.org/draft/2020-12/schema",
"type" : "object",
"properties" : {
"a" : {
    "type" : "integer",
    "format" : "int32",
    "description" : "First number"
},
"b" : {
    "type" : "integer",
    "format" : "int32",
    "description" : "Second number"
}
},
"required" : [ "a", "b" ],
"additionalProperties" : false
}], toolMetadata=DefaultToolMetadata[returnDirect=false]}

SyncToolSpecification[tool=Tool[name=add, description=Add two integers together, inputSchema=JsonSchema[type=object, properties={a={type=integer, format=int32, description=First number}, b={type=integer, format=int32, description=Second number}}, required=[a, b], additionalProperties=false, defs=null, definitions=null]], call=org.springframework.ai.mcp.McpToolUtils$$Lambda$1222/0x000000b801531130@762a10b6]

```
## 路由暴露
McpServerSseWebMvcAutoConfiguration

``` 
│  McpServerSseWebMvcAutoConfiguration                                        │
│      ↓ WebMvcSseServerTransportProvider (implements McpServerTransportProvider)│
│      │       .baseUrl("/")                                                  │
│      │       .sseEndpoint("/sse")                                           │
│      │       .messageEndpoint("/mcp/message")                              │
│      ↓                                                                      │
│      ↓ webMvcSseServerRouterFunction                                        │
│      │   = transportProvider.getRouterFunction()                            │
│      │   = RouterFunctions.route()                                          │
│      │       .GET("/sse",          this::handleSseConnection)   // SSE 长连接│
│      │       .POST("/mcp/message", this::handleMessage)         // JSON-RPC │
│      │       .build()                                                       │
│                                                                             │
│  请求流: Client → POST /mcp/message → handleMessage                         │
│         → McpSyncServer.handleToolCall → SyncToolSpecification.callHandler  │
│         → MethodToolCallback.call() → 反射 invoke Java 方法                 │
│         → CallToolResult → JSON-RPC Response → Client                      │


this.routerFunction = RouterFunctions.route()
    .GET(this.sseEndpoint, this::handleSseConnection)
    .POST(this.messageEndpoint, this::handleMessage)
    .build();

```

## 创建 McpSyncServer
```
本质 适配mcp sdk.
McpServerAutoConfiguration 
│  McpServerAutoConfiguration                                                 │
│      ↓ mcpSyncServer(ObjectProvider<List<SyncToolSpecification>> tools,     │
│      │               McpServerTransportProvider transportProvider)          │
│      ↓ McpServer.sync(transportProvider)                                   │
│      │       .serverInfo(name, version)                                     │
│      │       .capabilities(toolCapability)                                  │
│      │       .tools(toolSpecifications)    ← SyncToolSpec 列表注入          │
│      │       .build()                                                       │
│      ↓                                                                      │
│  McpSyncServer                                                              │
│      ├── tools: List<SyncToolSpecification>                                │
│      ├── 处理 tools/list → 返回 McpSchema.Tool 列表                         │
│      └── 处理 tools/call → 匹配 name → callHandler.apply(exchange, req)    │

McpServerSession.handle() 
    handleIncomingRequest(JSONRPCRequest request)
    handler = this.requestHandlers.get(request.method())  

requestHandlers.put(McpSchema.METHOD_TOOLS_CALL, toolsCallRequestHandler());
  │  反序列化 params → CallToolRequest
  │  遍历 this.tools 按 name 匹配
  │  tool.call().apply(exchange, arguments)
  MethodToolCallback.call()
```

## Mcp 交互

```
Client                              Server
  │                                    │
  │──── GET /sse ─────────────────────►│ handleSseConnection()
  │                                    │   uuid → sessionId
  │◄─── SSE: endpoint=/mcp/message ───│   返回 message endpoint
  │       ?sessionId=xxx               │
  │                                    │
  │──── POST /mcp/message ────────────►│ {"jsonrpc":"2.0","method":"initialize",...}
  │◄─── SSE: InitializeResult ────────│   协议版本 + capabilities 协商
  │                                    │
  │──── notifications/initialized ────►│
  │                                    │
  │──── tools/list ───────────────────►│ JSONRPCRequest[method=tools/list]
  │◄─── tools 列表 ───────────────────│ {tools:[{name:"add",...}]}
  │                                    │
  │──── tools/call ───────────────────►│ {"name":"add","arguments":{"a":17,"b":25}}
  │◄─── 结果 ─────────────────────────│ {"content":[{"type":"text","text":"42"}]}


 handleSseConnection
        {"jsonrpc":"2.0","method":"notifications/initialized"}

tool/list 
    JSONRPCRequest[jsonrpc=2.0, method=tools/list, id=2b03403d-1,params=PaginatedRequest[cursor=null]]

    JSONRPCResponse[jsonrpc=2.0, id=93969b11-1, result={tools=[{name=add, description=Add two integers together, inputSchema={type=object, properties={a={type=integer, format=int32, description=First number}, b={type=integer, format=int32, description=Second number}}, required=[a, b], additionalProperties=false}}]}, error=null]

llm 触发
    AssistantMessage [messageType=ASSISTANT, toolCalls=[ToolCall[id=, type=function, name=spring_ai_mcp_client_server1_add, arguments={"a":17,"b":25}]], textContent=<think>
    好的，用户让我把17和25相加。我需要检查提供的工具中有没有加法函数。看一下工具部分，确实有一个函数叫spring_ai_mcp_client_server1_add，它的功能是加两个整数a和b。参数是a和b，都是整数。用户给出的数字是17和25，所以应该直接调用这个函数，把它们的和作为结果。确认参数正确，然后生成对应的tool_call。
    </think>

/mcp/message?sessionId=6e18ca67-c431-4553-b4c6-5de3c5268227

    {"jsonrpc":"2.0","method":"tools/call","id":"aad59279-4","params":{"name":"add","arguments":{"a":17,"b":25}}}

```

keepAliveScheduler
每隔 N 秒向所有活跃 Session 发一个 JSON-RPC ping。 
  data: {"jsonrpc":"2.0","method":"ping","id":100}   ← 心跳


## Client 

McpClientAutoConfiguration
```
    ├── 读取 spring.ai.mcp.client.sse.connections.*
    │       ↓
    │   NamedClientMcpTransport("server1", HttpClientSseClientTransport)
    │
    ├── 创建 McpAsyncClient
    │       ↓ Initialize 握手
    │   JSONRPCRequest {
    │       method: "initialize",
    │       params: InitializeRequest {
    │           protocolVersion: "2024-11-05",
    │           clientInfo: { name: "spring-ai-mcp-client - server1", version: "1.0.0" }
    │       }
    │   }
    │
    ├── 注册 notification handlers:
    │       notifications/resources/list_changed
    │       notifications/tools/list_changed
    │       notifications/prompts/list_changed
    │
    └── AsyncMcpToolCallbackProvider
            ↓ listTools()
        为每个 tool 创建 AsyncMcpToolCallback
            工具名: spring_ai_mcp_client_{connectionName}_{toolName}
            例如:   spring_ai_mcp_client_server1_add

```


```
HttpClientSseClientTransport
| `FlowSseClient.SseEventHandler` | 处理 SSE 事件流 |

GET /sse HTTP/1.1
Accept: text/event-stream


# 1. 建立连接
SseEvent[type=endpoint, data=/mcp/message?sessionId=95deafc5-...]

# 2. 初始化响应
SseEvent[type=message, data={"jsonrpc":"2.0","id":"99a1734b-0",
    "result":{"protocolVersion":"2024-11-05",...}}]

# 3. 工具调用结果
SseEvent[type=message, data={"jsonrpc":"2.0","id":"9a9ca9ac-4",
    "result":{"content":[{"type":"text","text":"42"}],"isError":false}}]
```



## Streamable HTTP
HttpClientStreamableHttpTransport

```
Client                              Server (单端点 /mcp)
  │                                    │
  │──── POST /mcp ──────────────────►│  {"method":"initialize",...}
  │◄─── 200 JSON ───────────────────│  普通请求 → JSON 响应
  │     (Content-Type: application/json)
  │                                    │
  │──── POST /mcp ──────────────────►│  {"method":"tools/call",...}
  │◄─── 200 SSE stream ────────────│  长任务 → SSE 流式响应
  │     (Content-Type: text/event-stream)
  │                                    │
  │──── GET /mcp ───────────────────►│  (可选) 打开 SSE 流接收服务端通知
  │◄─── SSE: notifications ────────│  tools/list_changed 等
  │                                    │
  │   Header: Mcp-Session-Id: xxx     │  会话管理通过 Header 而非 URL 参数
  │   Accept: application/json,       │
  │           text/event-stream       │  客户端声明支持两种响应格式
```

# Agent flow 

本质是一个 Pipeline Engine。 

```
                 ┌───────────────────────┐
                 │        LLM             │
                 └──────────┬────────────┘
                            │
                        ChatModel
                            │
                    Advisor Pipeline
                            │
      ┌──────────────┬──────┴──────────┬───────────────┐
      │              │                 │               │
   Memory         RAG             ToolCall        Observability
      │              │                 │
      ▼              ▼                 ▼
ChatMemory      VectorStore        MCP Client
                                     │
                                     ▼
                                 MCP Server
                                     │
                                     ▼
                                 Tools
```

## ChatClient tool 调用方式
```java
    // 模式1: 带记忆 + 温度调节的 ChatClient
    this.chatClient = chatClientBuilder
            .defaultOptions(OllamaOptions.builder().temperature(0.9).build())
            .defaultAdvisors(PromptChatMemoryAdvisor.builder(messageWindowChatMemory).build())
            .build();

    // 模式2: 纯净 ChatClient（手动控制一切）， 控制tools调用
    this.pureChatClient = chatClientBuilder.build();

    // 模式3: 带 MCP 远程工具的 ChatClient
    this.mcpChatClient = chatClientBuilder
            .defaultToolCallbacks(toolCallbackProvider)
            .build();
```

自动调用tools
```
chatClient.prompt(query)
    .toolNames("getCityTimeTool")  
    .tools(new DateTimeTools())    
    .call().content();

mcpChatClient.prompt(query).call().content();
```

```
ChatClient.call()
    ↓
ChatModel.call(prompt) 
    ↓
DefaultToolExecutionEligibilityPredicate
    ↓ AssistantMessage 有 toolCalls ?
MethodToolCallback.call(arguments)    ← 反射调用 Java 方法
    ↓
ToolResponseMessage {
    responses: [ToolResponse{name, responseData}]
}
    ↓
再次调用 LLM（带 tool 结果）→ 最终回答
```

手动控制
```java
ToolCallingManager toolCallingManager = ToolCallingManager.builder()
        .toolExecutionExceptionProcessor(exceptionProcessor)
        .build();
ChatOptions chatOptions = ToolCallingChatOptions.builder()
            .toolCallbacks(ToolCallbacks.from(new DateTimeTools()))
            .internalToolExecutionEnabled(false)    // ← 关键：手动控制
            .build();
Prompt prompt = new Prompt(messages, chatOptions);
ChatResponse chatResponse = pureChatClient.prompt(prompt).call().chatResponse();

while (chatResponse.hasToolCalls() && iteration++ < maxIterations) {// 控制max 
    ToolExecutionResult toolResult = toolCallingManager
                .executeToolCalls(prompt, chatResponse);
    messages = chatMemory.get(conversationId);
    prompt = new Prompt(messages, chatOptions);
    chatResponse = chatClient.prompt(prompt).call().chatResponse();
    return chatResponse.getResult().getOutput().getText();
```


## Memory 和 Advisor


```
InMemoryChatMemoryRepository       ← 存储层（内存 Map） findByConversationId
        ↓
MessageWindowChatMemory            ← 策略层（滑动窗口，maxMessages=1000）  add(): 追加 + 裁剪超限消息
        ↓
PromptChatMemoryAdvisor            ← 拦截层（Advisor 模式，自动注入历史）
        ↓
ChatClient.defaultAdvisors(...)    ← 集成层
```

BaseAdvisor
```
	ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain);

	ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain);
```

PromptChatMemoryAdvisor
before
```
① 从 context 取 conversationId
② chatMemory.get(conversationId) → 读取历史消息
③ 过滤出 USER + ASSISTANT 消息，拼成文本：
     "USER: 你好\nASSISTANT: 你好！有什么..."
④ 用 systemPromptTemplate 渲染：
     "{instructions}
      Use the conversation memory from the MEMORY section...
      MEMORY:
      {memory}"
⑤ 替换 Prompt 中的 SystemMessage → augmentSystemMessage()
⑥ 当前 UserMessage 存入 chatMemory
```
after 
```
① 从 ChatResponse.getResults() 提取所有 AssistantMessage
② chatMemory.add(conversationId, assistantMessages) → 存入记忆
```

整体结构


```
请求方向 →                                              ← 响应方向

┌──────────────────────────────────────────────────────────────────┐
│ PromptChatMemoryAdvisor (order=1000)                             │
│   before(): 注入记忆 ──►                    ◄── after(): 存回复  │
│                         │                    │                   │
│   ┌─────────────────────▼────────────────────┴────────────────┐ │
│   │ RetrievalAugmentationAdvisor (order=0)                     │ │
│   │   before(): 检索文档拼上下文 ──►    ◄── after(): 附加文档   │ │
│   │                                │    │                      │ │
│   │   ┌────────────────────────────▼────┴───────────────────┐ │ │
│   │   │ ToolCallAdvisor (order 在中间)                       │ │ │
│   │   │   adviseCall(): do-while 循环                        │ │ │
│   │   │     ┌──────────────────────────────────────────────┐ │ │ │
│   │   │     │ ChatModelCallAdvisor (order=MAX, 链尾)        │ │ │ │
│   │   │     │   chatModel.call(prompt) → ChatResponse      │ │ │ │
│   │   │     └──────────────────────────────────────────────┘ │ │ │
│   │   │   有 toolCalls? → executeToolCalls → 再走一轮        │ │ │
│   │   └──────────────────────────────────────────────────────┘ │ │
│   └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## Rag
```
Pipeline本质

Query
  │
  ▼
Query Transform
  │
  ▼
Query Expand
  │
  ▼
Vector Retrieval
  │
  ▼
Document Join
  │
  ▼
Document PostProcess
  │
  ▼
Prompt Augment
  │
  ▼
LLM
```

### RetrievalAugmentationAdvisor RAG 管道架构

**本质 Pre-Retrieval -> Retrieval -> Post-Retrieval**。

**可以有很多策略优化**。 

#### QueryTransformer 链 
    
RewriteQueryTransformer 用 LLM 重写查询，使其更适合目标搜索引擎
TranslationQueryTransformer 用 LLM 翻译查询语言 
CompressionQueryTransformer 用llm Compressing conversation history and follow-up query into a standalone query

#### QueryExpander 
MultiQueryExpander 用llm 生成N维度 查询变体 
多角度扩展查询（默认 3 个变体）

#### DocumentRetriever（并发检索）
 VectorStoreDocumentRetriever
    vectorStore.similaritySearch(searchRequest) 
        embed(query) → HNSW → List<Document> 

```java
VectorStoreDocumentRetriever retriever = VectorStoreDocumentRetriever.builder()
        .vectorStore(esVectorStore)
        .similarityThreshold(0.5)      // 默认 0.0（全部返回）
        .topK(10)                      // 默认 4
        .filterExpression(expr)        // 
        .build();
```

ElasticsearchVectorStore
SimpleVectorStore
PgVectorStore
ChromaVectorStore

```
文档写入流程：
  List<Document>
       ↓
  EmbeddingModel.embed(doc.text)    ← 文本 → 向量 float[]
       ↓
  BatchingStrategy.batch()          ← 按 token 限制分批
       ↓
  VectorStore.add(documents)        ← 写入向量数据库
       ↓
  底层存储（ES BulkRequest）

检索流程：
  SearchRequest { query, topK, threshold, filter }
       ↓
  EmbeddingModel.embed(query)       ← 查询文本 → 查询向量
       ↓
  VectorStore.similaritySearch()    ← KNN 搜索
       ↓
  List<Document> [score, text, metadata]
```


#### DocumentJoiner
   去重，排序

#### DocumentPostProcessor

Post-Retrieval — Document Post-Processing 
  本质检索出来文档处理， Re-ranking， 文档压缩（摘要提取）Compression，业务Filtering
  很多事情可以做


#### QueryAugmenter 附加到docs 到 llm augmented Query
 originalQuery + documents → augmented Query

```
before(ChatClientRequest) {
    ① Query originalQuery = Query(userText, history, context)
    ② for (transformer : queryTransformers)          // 链式 transform
           transformedQuery = transformer.apply(q)
    ③ List<Query> expanded = queryExpander.expand(q)  // 扩展（可选）
    ④ Map<Query, List<Doc>> docs = expanded           // 并行检索
           .parallelStream()
           .map(q -> CompletableFuture(retriever.retrieve(q)))
    ⑤ List<Doc> joined = documentJoiner.join(docs)    // 合并去重
    ⑥ for (postProcessor : documentPostProcessors)    // 后处理
           joined = postProcessor.process(q, joined)
    ⑦ Query augmented = queryAugmenter.augment(q, joined)  // 增强
    ⑧ return chatClientRequest.mutate(augmentedPrompt)
}
```


### ES HNSW


HNSW（Hierarchical Navigable Small World）O(log N)  
  ANN = Approximate Nearest Neighbors（近似最近邻） 的一种，不看所以，快速查找。

```
Layer 3 (最稀疏):    A ─────────────────── D          少量"高速公路"
                     │                     │
Layer 2:             A ──── C ──── D ──── F            中等连接
                     │      │      │      │
Layer 1:             A ─ B ─ C ─ D ─ E ─ F ─ G        密集连接
                     │   │   │   │   │   │   │
Layer 0 (最密集):    A B C D E F G H I J K L M ...    所有节点都在

层级分配：随机（概率指数递减，和跳表一样）
邻居选择：计算实际向量距离（cosine/L2），贪心选最近的 M 个
ES默认 cosine。 
num_candidates 每个分片上的候选数 1.5 * topK, 分片聚合

```

# Langchain4j
```
MessageWindowChatMemory

DefaultRetrievalAugmentor
 EmbeddingStoreContentRetriever
 EmbeddingModel

类似 langchain 有chain组装固定的概念。  
9
Assistant assistant = AiServices.builder(Assistant.class)
     .chatModel(ollamaModel)
     .chatMemory(chatMemory)
     .contentRetriever(contentRetriever)
     .build();
    动态代理生成agent：    Object proxyInstance = Proxy.newProxyInstance(
flow固定在 DefaultAiServices.
```
大致差不多。