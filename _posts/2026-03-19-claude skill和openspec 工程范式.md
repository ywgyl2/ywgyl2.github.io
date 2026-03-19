---
title: Claude Skill/OpenSpec 工程范式
date: 2026-03-19 22:00:00 +0800
categories: [Claude Skill, Agent, OpenSpec]
tags: [agent, skill, cursor, claude, openspec]
pin: true
---

# Backgroud 
   之前研究实践 Claude Skill和OpenSpec总结，他们本质上都是一种 Agent的工程范式。 


# Skills

Skill本质是 一种Agent的标准工程范式。自带智能路由 route 加载特性，渐进披露来压缩context。 

Skills = 把“你会怎么想、怎么做”封装成 AI 可反复执行的能力单元。

高效的方式应该是 skill 包含 可执行的 CLI 工具,skill 内置 Python 脚本，可直接调用，减少token消耗。

渐进式披露，通过meta信息 路由skills，比mcp tools spes 浪费context的Agent 范式 要好很多。 

skillsmp.com 市场。

## Claude Code Skill
```
做了一个 repo-architect skill，用来分析项目的架构和核心流程。 Claude Code在大型repo分析上比cursor要强。 

1. 让Claude Code先用 Explore agents并行分析了解项目，生成CLAUDE.md （长期记忆）
2. 让Claude Code分析项目架构，核心模块和流程，生成md。
3. 让Claude Code使用 excalidraw-diagram skill生成靓图（美观，清晰）。

https://excalidraw.com/
```

分析OpenClaw的图，很清晰：

![OpenClaw 架构图](/assets/img/OpenClaw_Arch.png)

subagent 独立上下文，进程并行。skill主对话上下文，轻型sop flow的 Agent范式。

## OpenSkill + Cursor 

openskills install anthropics/skills
openskills sync 核心是同步到 cursor 的AGENTS.md <skills_system> <available_skills>， cursor能根据输入和skill元信息 调用 skill （openskills read <skill-name> 读取详细信息），也属于渐进式披露。

## Commands 差异
`.claude/commands/xx.md`
.cursor/commands/xx.md 是 `/xx` 显式调用，单文件 prompt。 不如 skill 可以附带 scripts，强大很多。 

## Skill结构
```
之前做了几个skills，1个是从 trello 做summary的skill，因为script一次生成了，用Playwright抓取list来总结，配合command 会很快。  

.claude/skills/trello-summary/
├── SKILL.md              # 必须：frontmatter(name, description) + 指令
├── scripts/
│   └── fetch_trello.py   # 可执行脚本
├── references/           # 可选：API 文档、参考
├── assets/               # 可选：模板、配置
└── requirements.txt      # 可选：Python 依赖
```


# OpenSpec

```
OpenSpec 其实也是一种 Spec-Driven Agent的工程范式，使用CLI工具链。 

OpenSpec 是一个「规范驱动（Spec-Driven）」的 CLI 工具链
把“规范”变成 AI 可稳定执行的输入源。
Specs 是真相（Single Source of Truth），Changes 是提案。 
最主要是 维护 project.md 和 spec.md（长期记忆）。 

openspec/
├── project.md              # 项目约定和规范
├── specs/                  # 当前真相（已构建的功能）
│   └── [capability]/
│       ├── spec.md         # 需求和场景
│       └── design.md       # 技术设计
├── changes/                # 提案（计划改变）
│   ├── [change-name]/
│   │   ├── proposal.md     # 为什么、改什么
│   │   ├── tasks.md        # 实施清单
│   │   ├── design.md       # 技术决策
│   │   └── specs/          # Delta 变更
│   │       └── [capability]/
│   │           └── spec.md
│   └── archive/            # 已完成的变更


project.md很重要，描述项目背景、技术栈、架构约定和命名规则。 

AGENTS.md – AI 的操作手册，指导 AI 工具理解 OpenSpec 的工作流程和目录结构，需要强制输出design.md需要更新这里。 

Proposal 处理流程（三阶段工作流） 非常清晰

阶段 1：创建变更提案（Creating Changes）
   tasks Plan的分解很重要
阶段 2：实现变更（Implementing Changes）
阶段 3：归档变更（Archiving Changes）

/openspec-proposal
/openspec-apply
/openspec-archive 总结
```

