---
title: Claude Code和Codex MultiAgents 初步理解
date: 2026-03-31 22:00:00 +0800
categories: [Claude Code, Codex, MultiAgent]
tags: [agent, claude, codex]
pin: true
---

## Backgroud 
   Claude Code部分公开，好奇心初步理解了Claude Code和开源Codex的 MultiAgent这块的设计, 思考AgentOS的要素。
 

## Claude Code

Claude Code 已知 包含几种multiAgent模式，其中Async Background大致和OpenClaw相似。
```text
┌──────────────────────────────┐
│ 主 agent 
│ 角色：coordinator            │
└──────────────┬───────────────┘
               │
               │ 1. 用户提出任务
               v
      [主 agent 理解任务并决策]
               │
               │ 2. 需要委派
               v
      AgentTool / spawn / runAgent
               │
               │ 3. 创建新的 agent 运行单元  （Plan/Expore/Verify）
               v
┌──────────────────────────────┐
│ 子 agent query() 实例        │
│ - 独立 messages              │
│ - 独立工具调用循环           │
└──────────────┬───────────────┘
               │
               │ 4. 子 agent 自己跑
               │    assistant -> tool_use -> tool_result -> assistant
               │
               ├───────────────┐
               │               │
               │ 5a. 普通 worker 完成
               │               │
               v               v
    <task-notification>      mailbox / teammate message
    回到主线程消息队列          回到 leader / teammate 通信层
               │               │
               └───────┬───────┘
                       │
                       │ 6. 主 agent 下一轮读到结果
                       v
┌──────────────────────────────┐
│ 主 agent                       │
│ - 收到结果后继续综合/分派    │
│   还是 spawn fresh           │
└──────────────────────────────┘
```
```
MultiAgent的几种模式：

Async Background模式 是 fire-and-forget，async 协程，和OpenClaw auto-announce很像，不过是task-notification机制（processQueueIfReady，加到/拉起下一轮ReActLoop  query() ）。

Fork Subagent模式 runForkedAgent 后台agent模式，共享 prompt cache（ **prompt cache hit rate**），同时避免污染父上下文，不需要回结果。
teammate 模式。

 In-Process Teammate（进程内队友）,Agent长期async 协程存在。通过 mailbox 轮询等待下一个 prompt（500ms 间隔）。
 Pane Teammate（终端面板队友：tmux / iTerm2），全新的 Claude Code 操作系统进程。
   都是通过 文件系统 mailbox（JSON 文件读写）来回传结果。 
   重任务 / 阻塞任务（CPU / IO），避免卡住同一个 event loop。
```


```text
                        ┌─────────┐
                   ┌───▶│ Explore  │──(只读搜索)──┐
                   │    │ (haiku)  │  同步阻塞     │ tool_result
                   │    └─────────┘              │ 
   用户 ──▶ 主 Agent │    ┌─────────┐              │
            (    ) ├───▶│  Plan   │──(只读规划)──┘
            唯一写者  │    │(inherit)│  同步阻塞       │
                   │    └─────────┘              ▼
                   │                      主 Agent 自己写代码
                   │    ┌─────────────┐          │
                   └───▶│Verification │◀─────────┘
                        │  (inherit)  │  后台异步 (background: true)
                        │ Bash 可执行 │──(只读验证)──▶ VERDICT (PASS/FAIL + 原因)
                        └─────────────┘     │
                                      task-notification
                                      (入 commandQueue)
  通信: commandQueue，子→父 单向
  通过AgentTool ->runAgent，runAgent.tx->query.ts。
      subAgent runAsyncAgentLifecycle，跑完enqueueAgentNotification。

  默认是主 Agent写，Coordinator 模式，主 Agent 会用 worker来写。 
```


## Codex

### ReAct Loop
```
run_turn(). tasks/regular.rs codex.rs::run_turn 标准的ReAct Loop.
```
```text
[准备上下文]
   |
   v
[向模型发起 sampling request]
   |
   +--> assistant message ----------> 记录输出 ----------+
   |                                                    |
   +--> reasoning item -----------> 记录/展示 ----------+
   |                                                    |
   +--> tool call ----------------> 执行工具 -----------> tool output 回注入
   |                                                    |
   +--> pending mailbox/input ----> 合并到下一轮 -------+
   |                                                    |
   +--> token 超限且仍需 follow-up -> compact ----------+
   |                                                    |
   +--> 无 follow-up ----------------------------------> turn complete
```

```
Mailbox 用来agents间通信。每个 Session（每个 agent ）各持一个 `Mailbox`（发送端）+ `MailboxReceiver`（接收端），取出msgs放到 下一个loop中。
Claude Code 的 teammate mailbox 是**文件系统 JSON 读写**。
```

### MultiAgents
```
   Codex::spawn()的subAgent是async task，能够在线程池(基于tokio)中并行跑。 Claude code则是协程（Node.js单线程 event loop），需要在eventLoop中调度。
   常用的也是 Supervisor这种模式。
```

```text
                        ┌──────────┐
                   ┌───▶│ Explorer │──(只读侦察)──┐
                   │    │  (  A )  │              │ 完成 → completed status
   用户 ──▶ Root Agent   └──────────┘              │ 通知 → mailbox
            主力写代码│    ┌──────────┐              │
            +协调者  ├───▶│ Explorer │──(只读侦察)──┤
                   │    │  (B  )   │              ▼
                   │    └──────────┘        Root Agent (wait_agent / 被动通知)
                   │                        读取结果 → 继续做本地工作或分配任务
                   │                              │
                   │  ┌──────────┐ ┌──────────┐ ┌──────────┐
                   └─▶│ Worker   │ │ Worker   │ │ Worker   │
                      │(   1  ) │ │(   2  )  │ │(  3   )  │
                      │handler.rs│ │server.rs │ │ test.rs  │
                      └──────────┘ └──────────┘ └──────────┘
                         tokio 异步真并行
  Worker 子 agent 真正写代码，Root 只做协调
  通信: tokio mpsc channel，任意 agent 间双向

  core/src/agent/role.rs built-in roles
  multi_agents_v2 spawn_agent， format_subagent_notification_message 给父agent发mailbox.


```


## AgentOS

强大的Memory层 + Tool/Skill/Plugin层 + Agents Matrix层。
  Context Engineering, Safety & Governance.
```text
┌───────────────────────────   Agent OS   ───────────────────────────┐
│                                                                       │
│  ┌─────────────────────── Scheduler ───────────────────────┐       │
│  │  event→ 选择 agent → 分配并发模式 → 生命周期管理             │       │
│  └──────────────────────────┬─────────────────────────────────┘       │
│                             ▼                                         │
│  ┌───────────────── Execution Layer ──────────────────┐               │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐   │               │
│  │  │  Agent A   │  │  Agent B   │  │  Agent C   │   │               │
│  │  │ ReAct Loop │  │           │  │           │   │               │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │               │
│  └────────┼───────────────┼────────────────┼──────────┘               │
│           ▼               ▼                ▼                          │
│  ┌───────────────── Communication Layer ──────────────┐               │
│  │   Mailbox / Channel / Event Stream                 │               │
│  │   (agent ↔ agent / agent ↔ scheduler)              │               │
│  └──────────┬─────────────────────────┬───────────────┘               │
│             ▼                         ▼                               │
│  ┌──────────────────┐      ┌─────────────────────┐                    │
│  │   Memory System   │      │  Tool/Skill System  │                    │
│  └──────────────────┘      └─────────────────────┘                    │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```