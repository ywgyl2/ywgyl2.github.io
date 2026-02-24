---
title: vLLM & SGLang Prefix Caching 对比
date: 2026-02-21 10:00:00 +0800
categories: [LLM, Inference]
tags: [vllm, sglang, kv-cache, prefix-caching]
pin: true
---

# Background
2025.10 之前总结的 vllm和sglang 核心kv cache数据结构 和 prefix-caching特点。 

Agent 多请求 共享system prompt，system prompt往往还很长，prefix-caching 就很有用，了解 vllm 和 sglang时，对比下他们差异和实现。 

共享同一段 system prompt：

| 请求 | 内容 |
|------|------|
| System Prompt | `You are a helpful AI assistant specialized in answering technical questions about programming and software engineering.` (26 tokens) |
| Request A | System Prompt + `What is AI?` (共 30 tokens) |
| Request B | System Prompt + `Tell me a joke.` (共 31 tokens) |


---

# vLLM KV Cache 结构

## PagedAttention

vLLM 采用  **PagedAttention** 机制，灵感来自操作系统虚拟内存分页。核心管理由 `BlockPool` 和 `KVCacheManager` 两个类协作完成，涉及多个关键数据结构：

```
blocks: list[KVCacheBlock],物理 block 的主列表，全局唯一的 block 对象池，大小 = num_gpu_blocks。 类似操作系统的页表映射.

cached_block_hash_to_block: BlockHashToBlockMap，哈希表，用于 prefix caching 的 O(1) 查找 。 哈希 满块时计算，用于缓存查找。 block_hash = hash(parent_block_hash + block_tokens + extra_hashes) extra_hashes涉及LoRA ID等扩展。 **链式依赖** 确保prefix 复用。 更新ref_cnt。 


free_block_queue: FreeKVCacheBlockQueue  空闲的blocks. 分配时从头部取，释放时插入尾部.

KVCacheManager#req_to_blocks: defaultdict[str, list[KVCacheBlock]] 记录每个请求当前持有哪些 blocks。  

本质是reqs的tokens kvcache 变长，切blocks存储，按照hash能复用 从前开始（prefix）的部分blocks。  

```
## 请求查找复用流程（`get_computed_blocks`）

```
新请求到达，token 序列: [t0, t1, t2, ..., tN]
                │
                ▼
   按 block_size 切分为若干 block_hash
   H0 = hash(∅ + tokens[0:16])
   H1 = hash(H0 + tokens[16:32])
   H2 = hash(H1 + tokens[32:48])
   ...
                │
                ▼
   逐个在 cached_block_hash_to_block 中查找
   ┌──────────────────────────────────┐
   │  H0 → 命中! block_id=3, 复用     │ 
   │  H1 → 命中! block_id=7, 复用     │ 
   │  H2 → 未命中, 停止匹配           │
   └──────────────────────────────────┘
                │
                ▼
   前 2 个 block 复用 (跳过 prefill)
   从第 3 个 block 开始分配新块:
     free_block_queue.popleft() → 取空闲 block
                │
                ▼
   更新 req_to_blocks[request_id] = [block_3, block_7, new_block, ...]
```

只有满块才会被缓存，部分填充的 block 不参与哈希匹配。

## 示例执行过程

### Request A 到达

```
System Prompt (26 tokens) + "What is AI?" (4 tokens) = 30 tokens

Block 0: tokens[0:16]   → 16/16, 满块 ✓
Block 1: tokens[16:30]  → 14/16, 部分块
```

- Block 0 **满块** → 计算哈希 `H0 = hash(∅ + tokens[0:16])`，写入 `cached_block_hash_to_block`
- Block 1 只有 14 tokens，不满 16 → **部分块，不缓存**

26 tokens 的 system prompt 只有前 16 tokens 被缓存在 Block 0 中。tokens[16:25]（system prompt 的后半段）和 tokens[26:29]（用户问题）混在 Block 1 里，因为不满块所以不缓存。


### Request B 到达

```
System Prompt (26 tokens) + "Tell me a joke." (5 tokens) = 31 tokens
```

1. 逐 block 计算哈希，查表匹配：
   - `H0 = hash(∅ + tokens[0:16])` → **命中！复用 Block 0**，`ref_cnt: 1 → 2`
2. 剩余 tokens 分配新 block：
   - Block 2: tokens[16:31] → 15/16, 部分块，不缓存

### 最终状态

```
Cache Blocks Map:
  H0(system_prompt 前 16 tokens) → Block 0  (ref_cnt=2)

Request Blocks:
  Request A → [Block 0, Block 1]
  Request B → [Block 0, Block 2]
```

```
              Request A                    Request B
              ┌─────────┐                  ┌─────────┐
              │ Block 0  │◄────共享────────►│ Block 0  │
              │ (cached) │                  │ (cached) │
              ├─────────┤                  ├─────────┤
              │ Block 1  │                  │ Block 2  │
              │ (partial)│                  │ (partial)│
              └─────────┘                  └─────────┘
```

本质上 vLLM prefix caching 的复用粒度受 block_size 限制。 粒度要粗一些。 

# SGLang KV Cache 结构

## RadixAttention

SGLang 采用 **RadixAttention**，用 **Radix Tree（基数树）** 管理 KV Cache。 本质上能做到 token级别的复用。 

```
RadixCache: 整体缓存
  Root (id=0, lock_ref=0)
  └─ Node1 (id=1): system prompt, 26 tokens
       │  key = [tok_0, tok_1, ..., tok_25]
       │  value = tensor([0, 1, 2, ..., 25])    ← GPU 槽位号
       ├─ Node2 (id=2): "What is AI?", 4 tokens
       │    key = [tok_26, tok_27, tok_28, tok_29]
       │    value = tensor([26, 27, 28, 29])     ← GPU 槽位号
       └─ Node3 (id=3): "Tell me a joke.", 5 tokens
            key = [tok_26, tok_27, tok_28, tok_29, tok_30]
            value = tensor([30, 31, 32, 33, 34])  ← GPU 槽位号

TreeNode.value 存的是 GPU 槽位号（整数索引）。实际的 KV 数据在 k_buffer[layer][槽位号] 和 v_buffer[layer][槽位号] 中。

MHATokenToKVPool: GPU 上实际的 KV 存储池，按层，根据显存来的 pool_size 是所有请求所有树节点共用的 token 槽位总数。

k_buffer: List[torch.Tensor]    长度 = layer_num
v_buffer: List[torch.Tensor]

k_buffer[0].shape = [pool_size, num_heads, head_dim]  # 第 0 层 Key, 
v_buffer[0].shape = [pool_size, num_heads, head_dim]  # 第 0 层 Value


通过槽位号寻址：k_buffer[layer=5][slot=25]  → shape [num_heads, head_dim]，即第 5 层，system prompt 第 26 个 token 的 Key 向量（32 个头，每头 128 维）
本质 token 级寻址，每个 token 独立一个槽位。 


ReqToTokenPool (请求 → token 映射表)


req_to_token: [max_reqs, max_context_len] 的二维 int32 tensor
请求活跃时的快照（以 Request B 为例）：
req_to_token[req_pool_idx=1] = [0,1,2,...,25, 30,31,32,33,34, 0,0,...]
                               └─ 前缀复用 ─┘ └─ 新分配 ──┘  └ 未用 ┘

前 26 个位置: 槽位 0~25，来自 Node1.value（prefix match 复用）
位置 26~30:   槽位 30~34，新分配（Tell me a joke. 的 KV）
位置 31+:     未使用，填充 0
```

## 前缀匹配流程（`match_prefix`）

```
新请求到达，token 序列: [t0, t1, t2, ..., tN]
                │
                ▼
   从 root_node 开始，逐 token（或逐 page）遍历树
   ┌─────────────────────────────────────────────┐
   │  root → child_key 匹配 tokens[0:26]         │
   │    → Node1 命中! 复用 Node1.value (26 slots) │
   │  Node1 → 无子节点匹配 tokens[26:]            │
   │    → 停止，返回 last_node = Node1            │
   └─────────────────────────────────────────────┘
                │
                ▼
   Req.prefix_indices = Node1.value  (26 个 KV 索引)
   Req.last_node = Node1
   剩余 tokens[26:] 需要 prefill，分配新 GPU 槽位
                │
                ▼
   prefill 完成后，insert() 将新 tokens 插入树
   → 创建新 TreeNode 作为 Node1 的子节点
```
本质 vLLM 匹配是 O(1) 哈希查找（但只能匹配满块），SGLang 匹配是 O(k) 树遍历（但能匹配任意长度前缀，且节点会自动分裂对齐边界）。

## 示例执行过程

### Request A 到达

```
System Prompt (26 tokens) + "What is AI?" (4 tokens) = 30 tokens
```

构建树：

```
Root
 └─ Node1: [system_prompt, 26 tokens]  → KV cache 存储
      └─ Node2: [What, is, AI, ?]       → KV cache 存储
```

### Request B 到达

```
System Prompt (26 tokens) + "Tell me a joke." (5 tokens) = 31 tokens
```

1. 从根遍历，逐 token 匹配：
   - tokens[0:26] → 与 Node1 完全匹配！**复用 Node1 的全部 26 tokens KV Cache**
   - tokens[26:31] → 无匹配，创建新分支

2. 树生长为：

```
Root
 └─ Node1: [system_prompt, 26 tokens]          ← 共享节点，完整复用
      ├─ Node2: [What, is, AI, ?]               ← Request A 独有
      └─ Node3: [Tell, me, a, joke, .]          ← Request B 独有
```

### 最终状态

```
RadixTree:
  Root
   └─ Node1: tokens[0:26], ref_cnt=2, KV_cache ✓    ← 26 tokens 全部复用
        ├─ Node2: tokens[26:30], ref_cnt=1, KV_cache ✓  (Request A)
        └─ Node3: tokens[26:31], ref_cnt=1, KV_cache ✓  (Request B)
```

Radix Tree 的关键优势：**节点会自动分裂**。 本质 **天然的prefix caching**. 


# 核心差异对比

本质缓存索引结构，一个用树、一个用哈希。
Prefix caching 查找,  O(1) 哈希查找 、 O(k) 树遍历（k = 前缀长度） 。

精度不同，Sglang 缓存命中率更高：

```
block_size = 16, system prompt = 26 tokens

vLLM:   只能复用前 16 tokens（1 个满块），剩余 10 tokens 重算  → 复用率 61%
SGLang: 全部 26 tokens 精确匹配复用                           → 复用率 100%
```