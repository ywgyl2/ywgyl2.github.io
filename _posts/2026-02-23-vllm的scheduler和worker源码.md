---
title: vllm scheduler worker源码
date: 2026-02-23 10:00:00 +0800
categories: [LLM, Inference]
tags: [vllm, scheduler, worker]
pin: true
---

# Flow

```
EngineCore.step()
    └─ scheduler.schedule() → SchedulerOutput
    └─ model_executor.execute_model(scheduler_output)   # Executor 分发到 多进程 Worker
         └─ worker.execute_model(scheduler_output)      # Worker 入口
              └─ model_runner.execute_model(...)        # GPUModelRunner 核心 把 SchedulerOutput 转成张量，驱动 model.forward / CUDA Graph
                   ├─ finish_requests / add_requests / update_requests
                   ├─ prepare_inputs(scheduler_output, num_tokens_after_padding)
                   ├─ model(input_ids, positions, inputs_embeds) 或 cudagraph.replay()
                   └─ execute_model_state = (hidden_states, input_batch, kv_connector_output)
    └─ model_executor.sample_tokens(grammar_output)     # 采样
         └─ model_runner.sample_tokens(grammar_output)
```

# 数据结构


```python
class Request:
    # ── 身份与参数────────────────────────────────────────────────────────
    request_id: str               # 全局唯一 ID
    arrival_time: float           # 到达时间（用于 FCFS 排序）
    priority: int                 # 优先级（PRIORITY 策略时用）
    sampling_params: SamplingParams  # temperature, top_p, max_tokens...
    max_tokens: int               # 本次最多生成多少 token

    # ── Token 状态（随执行动态更新）──────────────────────────────────────
    prompt_token_ids: list[int]   # 原始 prompt token ids（不变）
    num_prompt_tokens: int        # prompt 长度
    _output_token_ids: list[int]  # 已生成的 output tokens（每步 append）
    _all_token_ids: list[int]     # prompt + output 的完整序列
    num_computed_tokens: int      # 当前已完成计算的 token 数（每步调度后更新）
    spec_token_ids: list[int]     # 投机解码的 draft tokens

    # ── KV Cache 状态 ─────────────────────────────────────────────────────
    block_hashes: list[BlockHash] # 每个 block 的哈希（用于前缀缓存命中判断）
    num_cached_tokens: int        # 前缀缓存命中的 token 数（-1 表示尚未查询）
    num_external_computed_tokens: int  # KVConnector 远程缓存命中数（P/D 解耦） （Prefill/Decode Disaggregation）架构中的远程 Prefill 节点 tokens数

    # ── 调度状态 ─────────────────────────────────────────────────────────
    status: RequestStatus         # 当前状态（WAITING / RUNNING / PREEMPTED / ...）
    num_preemptions: int          # 被抢占次数（影响优先级策略）
    num_output_placeholders: int  # 异步调度时的 draft token 占位数
```

```python
# t=0: 用户发送 "What is vLLM?"
req = Request(
    request_id="req_42",
    prompt_token_ids=[1024, 374, 220, 85, 4312],  # 5 tokens
    sampling_params=SamplingParams(temperature=0.7, max_tokens=128),
)
req.status = RequestStatus.WAITING           # 刚加入
req.num_computed_tokens = 0
req._all_token_ids = [1024, 374, 220, 85, 4312]  # 仅 prompt

# t=1: schedule() 选中，prefill 5 tokens
req.status = RequestStatus.RUNNING
# _update_after_schedule: req.num_computed_tokens += 5 → 5

# t=2..N: decode 阶段，每步 +1 token
# step 2: req.num_computed_tokens += 1 → 6
#          req._output_token_ids = [2048]
#          req._all_token_ids = [1024, 374, 220, 85, 4312, 2048]
# step 3: num_computed_tokens → 7, _output_token_ids = [2048, 156]
# ...

# t=M: check_stop() 检测到 EOS 或 max_tokens 达到
req.status = RequestStatus.FINISHED_STOPPED
# _free_request(): KV blocks 释放, del requests["req_42"]
```

```
KVCacheBlock(block_id=5, ref_cnt=1,
             _block_hash=b'\xf3\x2a...\x01'  # hash([tok0..tok15])
             prev_free_block=None, next_free_block=None)
# → _block_hash 非 None，已插入 cached_block_hash_to_block 哈希表

KVCacheBlock(block_id=5, ref_cnt=2, _block_hash=b'\xf3\x2a...\x01',
             prev_free_block=None, next_free_block=None)
# → ref_cnt=2，只有两个请求都释放后才回到空闲队列

KVCacheBlock(block_id=5, ref_cnt=0, _block_hash=b'\xf3\x2a...\x01',
             prev_free_block=block_4, next_free_block=block_6)
# → ref_cnt=0 但仍在 cached_block_hash_to_block 中
# → 新请求来了可以直接命中（增加 ref_cnt），不用重新计算 KV
# → 若 BlockPool 空闲不足，LRU 驱逐会清除它（reset_hash + 回链表头）

```

```python
class BlockPool:
    num_gpu_blocks: int                        # GPU 上总 block 数（如 1024）
    blocks: list[KVCacheBlock]                 # 全部 block 的对象池，index = block_id

    free_block_queue: FreeKVCacheBlockQueue    # 空闲 block 的双向链表（LRU 顺序）
    cached_block_hash_to_block: BlockHashToBlockMap  # hash → block（前缀缓存查找表）
```

prefill和decode 混合
```python
SchedulerOutput(
    scheduled_new_reqs=[
        NewRequestData(req_id="req_B", prompt_token_ids=[101,202,303,404,505,606], ...)
    ],
    scheduled_cached_reqs=CachedRequestData(
        req_ids=["req_A"], resumed_req_ids=set(),
        new_block_ids=[None],           # req_A 未分配新 block
        num_computed_tokens=[30],       # req_A 已算 30 个 token
        num_output_tokens=[30], ...
    ),
    num_scheduled_tokens={"req_A": 1, "req_B": 4},  # A decode 1 token, B prefill 前 4 token
    total_num_scheduled_tokens=5,       # 1 + 4 = 5
    scheduled_spec_decode_tokens={},
    scheduled_encoder_inputs={},
    finished_req_ids=set(),
    free_encoder_mm_hashes=[],
)

 scheduled_spec_decode_tokens={
        "r1": [8821, 445, 1023],   # draft model 猜测的 3 个 token
        "r2": [592, 3001, 77],
        "r3": [2048, 156, 9999],
    },
```
InputBatch 和 CommonAttentionMetadata

```python
InputBatch(
    req_ids            = ["req_1", "req_2", "req_3"],
    num_reqs           = 3,
    idx_mapping        = tensor([0, 1, 2]),            # shape [3], batch_idx → req_state_idx
    num_scheduled_tokens = np.array([1, 1, 1]),        # shape [3]
    num_tokens         = 3,                            # 实际 token 数
    num_tokens_after_padding = 4,                      # pad 到 CUDA Graph size

    # ── 输入张量（GPU, bfloat16 / int32）──────────────────────
    input_ids    = tensor([8821, 445, 1023, 0]),        # shape [4],  最后 0 是 pad
    positions    = tensor([10,   25,  5,   0]),         # shape [4]
    query_start_loc = tensor([0, 1, 2, 3, 4]),          # shape [4]  (num_reqs+1)

    # ── Attention 元数据（每层共用同一份）────────────────────────
    attn_metadata = {
        "model.layers.0.self_attn": FlashAttentionMetadata(
            query_start_loc = tensor([0,1,2,3,4]),
            seq_lens        = tensor([11,26,6]),
            max_query_len   = 1,
            max_seq_len     = 26,
            block_table     = tensor([[blk_0,-1],[blk_5,blk_6],[blk_2,-1]]),
            slot_mapping    = tensor([blk_0*16+10, blk_6*16+9, blk_2*16+5, 0]),
            num_actual_tokens = 4,
            causal          = True,
        ),
        "model.layers.1.self_attn": <同上，同一对象>,  # 32 层共用
        ...
        "model.layers.31.self_attn": <同上>,
    }
```
RequestState
```
`req_states` 是 ModelRunner 上持久保存的请求状态表，用来算 idx_mapping， positions等。 

 req_id_to_index = {"req_1": 0, "req_2": 1, "req_3": 2}   ← req_id → 槽位下标
  index_to_req_id = {0: "req_1", 1: "req_2", 2: "req_3"}
  free_indices    = [3]                                       ← 空闲槽位
prompt_len = np.array([5, 8, 5, 0], dtype=int32)
  #                      ↑  ↑  ↑  ↑
  #              req_1的prompt长5  req_2的prompt长8  req_3的prompt长5  空
  num_computed_prefill_tokens = np.array([5, 8, 5, 0])
    # req_1/req_2/req_3 的 prefill 已全部算完（≤ prefill_len）

  last_sampled_tokens:  shape [4, 1]，GPU tensor
    tensor([[8821],   ← req_1 上一步采样到的 token
            [445],    ← req_2
            [1023],   ← req_3
            [0]])     ← 空槽（无效）
```
ForwardContext attn计算需要的数据，用 `attn_metadata` 和 `slot_mapping` 访问 Paged KV Cache。

```python
_forward_context = ForwardContext(
    # ── 核心：每层 Attention 需要的元数据 ────────────────────────────────
    attn_metadata = {
        "model.layers.0.self_attn":  FlashAttentionMetadata(...),  # 32 层同一对象
        "model.layers.1.self_attn":  FlashAttentionMetadata(...),
        ...
        "model.layers.31.self_attn": FlashAttentionMetadata(...),
    },

    # ── 每层 KV Cache 写入 slot（与 attn_metadata 配套）────────────────
    slot_mapping = {
        "model.layers.0.self_attn":  tensor([blk_0*16+10, blk_6*16+9, blk_2*16+5, 0]),
        ...
        "model.layers.31.self_attn": tensor([blk_0*16+10, blk_6*16+9, blk_2*16+5, 0]),
    },...
```

ModelRunnerOutput 返回给 Scheduler。
```python
ModelRunnerOutput(
    req_ids          = ["req_1", "req_2", "req_3"],
    req_id_to_index  = {"req_1": 0, "req_2": 1, "req_3": 2},

    sampled_token_ids = [
        [2048],    # req_1：本步生成 1 个 token（普通 decode）
        [156],     # req_2
        [9999],    # req_3
    ],

    logprobs              = None,   # 未开启 logprobs
    prompt_logprobs_dict  = {},     # 全是 decode，无 prefill logprobs
    pooler_output         = None,   # 非 embedding 模型
    kv_connector_output   = None,   # 单机，无跨节点 KV 传输
    cudagraph_stats       = CUDAGraphStat(num_tokens=4, graph_size=4),
)
```


# schedule 

调度参数。
```
max_num_running_reqs     = 256    (最多同时处理 256 条请求)
max_num_scheduled_tokens = 8192   (每步最多处理 8192 个 token)  token_budget
max_model_len            = 4096   (单请求最长序列 4096)
block_size               = 16     (每个 KV block 存 16 个 token)
num_gpu_blocks           = 1024   (GPU 显存中 1024 个 block，共 16384 个 slot)
```

```
schedule()
  │
  ├─ 初始化局部变量
  │   token_budget = max_num_scheduled_tokens  (e.g. 8192)
  │   scheduled_new_reqs = []
  │   scheduled_running_reqs = []
  │   preempted_reqs = []
  │   num_scheduled_tokens = {}   # req_id → 本步分配的 token 数
  │
  ├─ ═══ Phase 1: 优先处理 RUNNING 请求 ═══
  │   for request in self.running:
  │       ① 计算 num_new_tokens（还需要处理多少 token？）
  │       ② allocate_slots（尝试分配 KV block）
  │       ③ 若失败 → 抢占低优先级请求，循环重试
  │       ④ 若成功 → 加入 scheduled_running_reqs, token_budget -= num_new_tokens
  │
  ├─ ═══ Phase 2: 从 WAITING 队列调度新请求 ═══
  │   while self.waiting and token_budget > 0 and len(running) < max_running:
  │       ① peek 队头请求
  │       ② 状态过滤（FSM/远程KV/流式输入等特殊状态 → skip）
  │       ③ get_computed_blocks（查询前缀缓存命中）
  │       ④ 计算 num_new_tokens
  │       ⑤ allocate_slots（分配 KV block）
  │       ⑥ 若失败 → break（KV block 耗尽，停止接纳新请求）
  │       ⑦ 若成功 → pop 出 waiting，加入 running + scheduled_new_reqs
  │
  ├─ ═══ Phase 3: 构建 SchedulerOutput ═══
  │   new_reqs_data = [NewRequestData.from_request(req, block_ids) ...]
  │   cached_reqs_data = _make_cached_request_data(running_reqs, ...)
  │   scheduler_output = SchedulerOutput(...)
  │
  └─ ═══ Phase 4: 收尾更新 ═══
      _update_after_schedule(scheduler_output)
        → for each scheduled req: req.num_computed_tokens += num_scheduled_tokens
        → self.finished_req_ids = set()   (清空已完成集合)
```
```python
for request in self.running:
    # ── 计算本步需要处理多少 token ────────────────────────────
    num_new_tokens 涉及限制：
     Chunked Prefill 限制long_prefill_token_threshold
     token_budget
     max_model_len

#从 waiting 队列调度新请求
while self.waiting and token_budget > 0:
  if request.num_computed_tokens == 0: # 第一次调度该请求，查询前缀缓存
          new_computed_blocks, num_new_local_computed_tokens = (
            kv_cache_manager.get_computed_blocks(request)
            #根据 request.block_hashes（已在 add_request 时预计算） 在 BlockPool 的哈希表中查找最长匹配前缀
            #coordinator.find_longest_cache_hit
        )
        # 返回：已缓存的 block 列表 + 命中的 token 数
self.running.append(request)             # 加入 running
    scheduled_new_reqs.append(request)
    num_scheduled_tokens[request.request_id] = num_new_tokens
    token_budget -= num_new_tokens
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens = num_computed_tokens   # 设置前缀缓存命中数

_update_after_schedule中
 request.num_computed_tokens += num_scheduled_token

```
## prompt -> Request


```
┌─────────────────────────────────────────────────────────────────────┐
│  API Server 进程                            │
│                                                                     │
│  HTTP POST /v1/completions {"prompt": "What is vLLM?"}              │
│        │                                                            │
│        ▼                                                            │
│  openai_serving_completion.py                                       │
│        │  create_completion()                                       │
│        ▼                                                            │
│  AsyncLLM.generate()                                                │
│        │  ① add_request()                                           │
│        │      └─ InputProcessor.process_inputs()                    │
│        │              → tokenize("What is vLLM?") → [1024,374,...]  │
│        │              → 验证 sampling_params                         │
│        │              → 构造 EngineCoreRequest                       │
│        │      └─ engine_core.add_request_async(EngineCoreRequest)   │
│        │              └─ ZMQ PUSH → EngineCoreProc 进程             │
│        │                                                            │
│  ② output_handler (asyncio task)                                    │
│        └─ engine_core.get_output_async() ← ZMQ PULL 等待结果       │
└──────────────────────────────────────────────────────────────────────┘
                              │ ZMQ ipc:// socket（msgpack 序列化）
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EngineCoreProc 进程（后端，独立 Python 进程）                         │
│                                                                     │
│  input_thread (ZMQ recv)   process_input_sockets                                          │
│        └─ 接收 EngineCoreRequest → input_queue.put()                │
│                                                                     │
│  core_busy_loop (主线程)                                              │
│        └─ input_queue.get()                                         │
│              └─ preprocess_add_request(EngineCoreRequest)           │
│                      → Request.from_engine_core_request()            │
│                      → 计算 block_hashes（前缀缓存 hash）             │
│                           │
│              └─ EngineCore.add_request(request)                     │
│                      └─ scheduler.add_request(request)              │
│                              └─ waiting.append(request)  ← 进入队列 │
│                                                                     │
│        loop:                                                        │
│              step() → scheduler.schedule() → SchedulerOutput        │
│                     → execute_model() → ModelRunnerOutput            │
│                     → scheduler.update_from_output()                │
│                     → output_queue.put(EngineCoreOutputs)           │
│                                                                     │
│  output_thread (ZMQ send)                                           │
│        └─ output_queue.get() → ZMQ PUSH → API Server 进程           │
└─────────────────────────────────────────────────────────────────────┘
```

```
用户输入（字符串）
    │
    │  InputProcessor.process_inputs()
    │     tokenize()
    │
    ▼
EngineCoreRequest                           ← 可 msgpack 序列化，跨进程传输
    request_id        = "req-abc123"
    prompt_token_ids  = [1024, 374, 220, 85, 4312]
    sampling_params   = SamplingParams(temperature=0.7, max_tokens=128)
    eos_token_id      = 2
    arrival_time      = 1700000000.0
    mm_features       = None
    │
    │  ZMQ msgpack → EngineCoreProc 进程
    │  Request.from_engine_core_request()
    │  + 计算 block_hashes
    │
    ▼
Request                                     ← 带方法的调度对象，仅在 EngineCore 进程内
    request_id        = "req-abc123"
    prompt_token_ids  = [1024, 374, 220, 85, 4312]
    num_prompt_tokens = 5
    _all_token_ids    = [1024, 374, 220, 85, 4312]
    num_computed_tokens = 0
    status            = WAITING
    block_hashes      = [hash_block0]       ← 前缀缓存 hash 链
    arrival_time      = 1700000000.0
    │
    │  scheduler.add_request()
    │
    ▼
waiting 队列中等待 schedule() 调度
```



## prefix caching
```
系统已有 req_old 的缓存：
  req_old prompt = [128000, 9906, 499, 527, 1790, 1268, 1790, 220, ...]  (16 tokens)
  这 16 tokens 被 hash → block_hash_A，block_id=5 命中缓存

新请求 req_new 到达：
  req_new prompt = [128000, 9906, 499, 527, 1790, 1268, 1790, 220, 1148, 374, ...]
                    ← 前 16 tokens 与 req_old 完全相同 →
  block_hashes[0] = block_hash_A  (第一个 block 的哈希)
  block_hashes[1] = block_hash_B  (后续 block)

get_computed_blocks(req_new):
  find_longest_cache_hit([block_hash_A, block_hash_B], max=19)
  → 找到 block_id=5 命中 block_hash_A
  → num_new_local_computed_tokens = 16   (命中 1 个 block = 16 tokens)
  → computed_blocks = [KVCacheBlock(block_id=5)]

allocate_slots(req_new, num_new_tokens=4, num_new_computed_tokens=16,
               new_computed_blocks=[block_5]):
  → 只需要为 4 个新 token 分配 1 个 block（token 16~19 → 新 block_id=7）
  → block_5 的 ref_cnt 增加（共享），不重新分配

schedule() 中：
  request.num_computed_tokens = 0 + 16 = 16   (前缀缓存节省了 16 token 的计算)
  num_scheduled_tokens["req_new"] = 4           (本步只需 prefill 后 4 个 token)
  token_budget -= 4                             (节省了 16 的 token 配额！)
```

## allocate_slots
KVCacheManager.allocate_slots()

```python
def allocate_slots(request, num_new_tokens, num_new_computed_tokens=0,
                   new_computed_blocks=None, num_lookahead_tokens=0, ...):
    """
    块分配布局：
    ┌──────────────┬────────────────┬──────────────┬────────────────┐
    │  comp blocks │  new_comp blks │   ext_comp   │   new blocks   │
    │(已有, 不重分配)│(前缀缓存命中)   │(远程KV命中)   │(本步新分配)     │
    └──────────────┴────────────────┴──────────────┴────────────────┘
"""
  算total_tokens_needed 得 num_blocks_needed， block_pool.num_free_blocks够 
      new_blocks = block_pool.allocate(num_needed)
    return KVCacheBlocks(new_blocks)

分配失败 → 抢占 running[-1]（FCFS 最后加入的请求）   FCFSRequestQueue
    preempted_req = self.running.pop()      # 弹出末尾
    self._preempt_request(preempted_req)    # 释放其 KV blocks
         self.kv_cache_manager.free(request)   # ① 释放该请求持有的所有 KV block
        self.waiting.prepend_request(request)
    preempted_reqs.append(preempted_req)

PRIORITY 策略：抢占 running 列表中 priority 最低（数值最大）的请求

```

KV Block 的物理布局（Llama-7B，block_size=16）
```
GPU 显存中的 KV Cache（以 layer 0 为例）：

  key_cache  shape: [num_blocks, block_size, num_heads, head_dim]
             = [1024, 16, 32, 128]   → 约 256MB (fp16)
  value_cache shape: [1024, 16, 32, 128]

  Block 0 (物理 block_id=0):
    slots [0..15]: 存 16 个 token 的 K/V

  Block 1 (block_id=1):
    slots [16..31]: 另 16 个 token 的 K/V

  req_42 拥有 blocks [3, 7, 15]:
    token 0~15  → block_id=3, slots [48..63]
    token 16~31 → block_id=7, slots [112..127]
    token 32~47 → block_id=15, slots [240..255]

scatter kernel 要的是一个连续的 int 数组，不接受 (block_id, offset) 对。按索引散射写入。

```

```
结构（哨兵头尾节点，避免边界判断）：

  fake_head ↔ block_2 ↔ block_7 ↔ block_15 ↔ block_3 ↔ ... ↔ fake_tail
              ↑ LRU最久未用                               ↑ 最近释放的

  popleft() → 从 fake_head.next 取（LRU 优先驱逐最久未用的）
  append()  → 插到 fake_tail.prev（最近释放的在末尾）

  hash 的计算方式（chain hashing，每个 block 包含前序信息）
  block_0 的 hash = hash(NONE_HASH + tokens[0:16])
                   ↑ 随机初始 seed（跨进程可配置）

block_1 的 hash = hash(block_0_hash + tokens[16:32])
                   ↑ 把前一个 block 的 hash 链进来

block_2 的 hash = hash(block_1_hash + tokens[32:48])
                   ↑ 确保相同 block 内容但前缀不同时 hash 也不同
                  
```

```
BlockPool
  ├── blocks[1024]              ← 全部 KVCacheBlock 的对象池
  │     block_id=0  (null)
  │     block_id=1  ref_cnt=2, hash=0xABCD  ← 被两个请求共享
  │     block_id=2  ref_cnt=1, hash=None    ← 已分配，未填满
  │     block_id=3  ref_cnt=0, hash=None    ← 空闲
  │     ...
  │
  ├── free_block_queue (双向链表，LRU 顺序)
  │     fake_head ↔ [oldest_free] ↔ ... ↔ [newest_free] ↔ fake_tail
  │     num_free_blocks = 997
  │
  └── cached_block_hash_to_block (哈希表，前缀缓存命中)
        0xABCD      → block_id=1   (ref_cnt=2，正在被使用)
        0xDEF0      → block_id=5   (ref_cnt=0，空闲但可命中，LRU 驱逐候选)
        ...

KVCacheManager
  └── coordinator
        └── block_pool (即上面的 BlockPool)
        └── request_to_blocks: dict[str, list[KVCacheBlock]]
              "req_A" → [blk1, blk2]
              "req_B" → [blk1, blk2, blk3]   (blk1/blk2 与 req_A 共享)
```


# prepare_inputs

## Cuda Graph 

CUDA Graph 的本质是把一系列 GPU kernel 调用录制下来，之后直接 replay，跳过 Python 解释和 CUDA driver 开销。

启动时预录: graphs[1], graphs[2], graphs[4], graphs[8], graphs[16] ...
            shape固定↗     shape固定↗    shape固定↗
向上取整到最近的预录 size，多出来的位置用 padding token 填充。

[1, 2, 4, 8, 16, 24, 32, 40, 48, ..., 248, 256, 272, 288, ..., 496, 512]
 ↑小batch    ↑步长8                          ↑步长16

## 例子

例子
```
模型参数：Llama-7B
  hidden_size      = 4096
  num_layers       = 32
  num_heads        = 32 (num_kv_heads = 32, GQA=1)
  head_dim         = 128
  vocab_size       = 32000
  block_size       = 16

本步状态：
  req_1: 已生成 10 tokens, last_sampled_token = 8821
  req_2: 已生成 25 tokens, last_sampled_token = 445
  req_3: 已生成 5 tokens,  last_sampled_token = 1023
  → total_num_scheduled_tokens = 3
```
cudagraph_sizes 映射表: {1→1, 2→2, 3→4, 4→4, 5→8, ...} 查表: cudagraph_sizes[3] = 4

本质 "用少量冗余计算换 CUDA Graph 加速" 的代价。 

## 核心标记

```python
req_ids = ["req_1", "req_2", "req_3"]
idx_mapping = [0, 1, 2]                     # shape: [3]

# input_ids：3 个真实 token + 1 个 padding
input_ids = [8821, 445, 1023, 0]             # shape: [4]  
#            req_1  req_2 req_3  pad

# positions：每个请求当前位置 = num_computed_tokens
positions = [10, 25, 5, 0]                   # shape: [4] 
#            ↑req_1已算10个  ↑pad

# query_start_loc：每请求在 batch 中的 token 起始位置
query_start_loc = [0, 1, 2, 3, 4]           # shape: [num_reqs+1] = [4] 
#                  ↑r1 ↑r2 ↑r3 ↑pad/end

decode 和 prefill 混合的情况
→ query_start_loc = [0, 1, 2, 10]
                     ↑  ↑  ↑   ↑
                    B  C  A   end

含义：
  Req B 的 token 在 input_ids[0:1]    → 1 个 decode token
  Req C 的 token 在 input_ids[1:2]    → 1 个 decode token
  Req A 的 token 在 input_ids[2:10]   → 8 个 prefill token

# seq_lens：每请求的总序列长度（KV 历史 + 本步 token）
seq_lens = [11, 26, 6]                       # shape: [3]
#           10+1  25+1  5+1

# block_table  reqs的 blks
block_table = [[blk_0, -1],                  # shape: [3, max_blocks_per_seq]
               [blk_5, blk_6],
               [blk_2, -1]]

# slot_mapping：每 token 在 KV Cache 的物理 slot
# slot = block_table[req][pos // block_size] * block_size + pos % block_size
slot_mapping = [blk_0*16+10, blk_6*16+9, blk_2*16+5, 0]  # shape: [4]

# logits_indices：需要采样的 token 在 input_ids 中的位置
logits_indices = [0, 1, 2]                   # shape: [3]  (decode 时 = 每请求最后 1 个 token)
```

## model.forward()

`model.forward()`（CUDA Graph Replay）

```
input_ids                               [4]         (int32)  B=4
   ↓ embed_tokens
token_embeddings                        [4, 4096]   (float16/bfloat16)
   ↓
=== Layer 0 ~ 31（每层相同 shape）===
   ├─ input_layernorm
   │   hidden_states                    [4, 4096]
   <!-- ├─ self_attn
   │   ├─ Q = Wq @ hidden              [4, 32, 128] → reshape → [4, 32, 128]
   │   ├─ K = Wk @ hidden              [4, 32, 128]
   │   ├─ V = Wv @ hidden              [4, 32, 128]
   │   ├─ 写入 KV Cache（按 slot_mapping 写入物理 slot）
   │   │   K_cache[slot] = K            写入 4 个 slot（含 1 个 pad 的无效写入）
   │   │   V_cache[slot] = V
   │   ├─ Paged Attention（读取历史 KV + 当前 K/V）
   │   │   对 req_1: attend over 11 tokens (10 history + 1 current)
   │   │   对 req_2: attend over 26 tokens
   │   │   对 req_3: attend over 6 tokens
   │   │   对 pad:   无效计算，结果被丢弃
   │   └─ attn_output                   [4, 4096]
   ├─ post_attention_layernorm
   │   hidden_states                    [4, 4096]
   ├─ mlp (gate_up_proj → SiLU → down_proj)
   │   gate_up                          [4, 11008×2] → SiLU → [4, 11008]
   │   down                             [4, 4096]
   └─ residual add
       hidden_states                    [4, 4096]
=== 重复 32 层 ===

final_layernorm                         [4, 4096]
→ hidden_states                         [4, 4096]

``` -->

Graph replay 结束后：
hidden_states = cudagraph_manager.run(4)    # 内部: graphs[4].replay()
# → 返回 self.hidden_states[:4]            shape: [4, 4096]
```

```
prepare_inputs → 构造 InputBatch 使用处
     │
     ├─ execute_model
     │    ├─ set_forward_context(input_batch.attn_metadata, slot_mappings)
     │    └─ model(input_ids, positions)
     │         └─ 每层 Attention.forward()
     │              └─ get_forward_context().attn_metadata["layer_X"]  ← 从 context 取
     │                   → flash_attn_varlen_func(block_table, seq_lens, ...)
     │
     └─ sample_tokens(hidden_states, input_batch, grammar_output)
          ├─ hidden_states[input_batch.logits_indices]  ← 丢弃 padding
          ├─ model.compute_logits(sample_hidden_states)
          └─ sampler(logits, input_batch.req_ids, ...)
```

## flash_attn_varlen_func

```
kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
              = [2, N, 16, 32, 128]
                 ↑
                 0=key_cache, 1=value_cache

key_cache, value_cache = kv_cache.unbind(0)
key_cache:   [N, 16, 32, 128]    ← 所有物理 block 的 K
value_cache: [N, 16, 32, 128]    ← 所有物理 block 的 V

写入新KV 
对于 req_1（pos=10, slot=blk_0*16+10）：
  block_idx  = slot // block_size = blk_0
  offset     = slot % block_size  = 10

  key_cache[blk_0][10][:][:] = K_new[0]     # [32, 128] 写入 32 个 head
  value_cache[blk_0][10][:][:] = V_new[0]

req_1: seq_len=11, block_table=[blk_0, -1]
  → blk_0 中 slot 0~10 有效（11 个 token）

收集kv
以 head_0 为例，从 KV Cache 中逻辑上看到：

  K_all = [key_cache[blk_0][0],    ← pos 0 的 K, shape [128]
           key_cache[blk_0][1],    ← pos 1
           ...
           key_cache[blk_0][10]]   ← pos 10（刚写入）
  → K_all shape: [11, 128]        （seq_len=11 个历史+当前 K）

  V_all shape: [11, 128]           同理

Scaled Dot-Product Attention（每请求独立，每 head 独立）
以 req_1, head_0 为例：

Q = query[0, head_0, :]            shape: [128]     ← 1 个 query token
K = K_all                          shape: [11, 128] ← 11 个 key tokens
V = V_all                          shape: [11, 128] ← 11 个 value tokens
scale = 1 / sqrt(head_dim) = 1 / sqrt(128) = 0.0884

┌──────────────────────────────────────────────────┐
│  ① Attention Score                               │
│     S = Q @ K^T * scale                          │
│     [128] @ [128, 11] * 0.0884                   │
│     → S shape: [11]                              │
│     S = [s_0, s_1, s_2, ..., s_10]               │
│                                                  │
│  ② Causal Mask（decode 时无影响）                  │
│     decode 只有 1 个 query（pos=10）              │
│     pos_q=10 >= pos_k=0..10 全部成立              │
│     → 不 mask 任何 key（全部可见）                 │
│                                                  │
│  ③ Softmax                                       │
│     P = softmax(S)                               │
│     [11] → [11]  (概率分布，和为 1)               │
│     P = [p_0, p_1, ..., p_10]                    │
│                                                  │
│  ④ Weighted Sum                                  │
│     O = P @ V                                    │
│     [11] @ [11, 128]                             │
│     → O shape: [128]                             │
│     = p_0*V[0] + p_1*V[1] + ... + p_10*V[10]    │
└──────────────────────────────────────────────────┘

→ output[0, head_0, :] = O    shape: [128]

多 reqs并行： 

req_1 (decode, seq_len=11):
  Q: [1, 32, 128]  attend over K/V: [11, 32, 128]  → output: [1, 32, 128]
  每个 head: [128] @ [128,11] → softmax → [11] @ [11,128] → [128]

req_2 (decode, seq_len=26):
  Q: [1, 32, 128]  attend over K/V: [26, 32, 128]  → output: [1, 32, 128]
  每个 head: [128] @ [128,26] → softmax → [26] @ [26,128] → [128]

req_3 (decode, seq_len=6):
  Q: [1, 32, 128]  attend over K/V: [6, 32, 128]   → output: [1, 32, 128]
  每个 head: [128] @ [128,6]  → softmax → [6]  @ [6,128]  → [128]

pad (无效):
  Q: [1, 32, 128]  → 计算结果被丢弃

合并后: output shape = [4, 32, 128] → flatten → [4, 4096]
```


## sample_tokens()

```python

# 只取需要采样的 token（丢弃 padding）
sample_hidden_states = hidden_states[logits_indices]  # [4, 4096][[0,1,2]] → [3, 4096]

# 投影到词表
logits = model.compute_logits(sample_hidden_states)   # [3, 4096] @ [4096, 32000] → [3, 32000]

# 采样
sampled_token_ids = sampler(logits, sampling_params)   # [3]  每请求 1 个 token_id
# 例如: [2048, 156, 9999]
```




