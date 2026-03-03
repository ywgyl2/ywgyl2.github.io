---
title: vllm attention
date: 2026-02-23 10:00:00 +0800
categories: [LLM, Inference]
tags: [vllm, attention]
pin: true
---

# Flow
```
模型层（Llama / Mistral / ...）
  └─ Attention.forward(query, key, value)
       └─ unified_attention_with_output(q, k, v, output, layer_name)
            └─ impl.forward(query, key, value, kv_cache, attn_metadata, output)
                 └─ FlashAttentionImpl / FlashInferImpl / TritonAttentionImpl / ...
                      └─ flash_attn_varlen_func() / flashinfer_wrapper() / paged_attention_v1() / ...
```
FlashAttentionImpl
FlashInferImpl
TritonAttentionImpl
paged_attention_v2

# LlamaAttention 
```
hidden_states.shape = [12, 4096]    # [num_tokens, hidden_size]
positions.shape     = [12]          # [10, 25, 0,1,2,3,4,5,6,7, 0, 0]
                                    #  ↑r1 ↑r2  ↑── req_3 ──↑  ↑pad↑

# hidden_states 每一行是一个 token 的 4096 维表示向量
# 第 0 行: req_1 的 decode token（第 10 个位置）
# 第 1 行: req_2 的 decode token（第 25 个位置）
# 第 2~9 行: req_3 的 8 个 prefill tokens
# 第 10~11 行: padding（全 0）
```

QKV 合并投影，再切分，
```python
qkv, _ = self.qkv_proj(hidden_states)

# QKVParallelLinear 内部：
#   权重 W_qkv shape: [hidden_size, q_size + kv_size + kv_size]
#                    = [4096,        4096   + 1024    + 1024   ]
#                    = [4096, 6144]
#
#   qkv = hidden_states @ W_qkv^T
#       = [12, 4096] @ [4096, 6144]^T  → 一次 GEMM 搞定 Q/K/V！
#
#   qkv.shape = [12, 6144]

# 为什么合并？
#   分开做 3 次 GEMM: [12,4096]@[4096,4096] + [12,4096]@[4096,1024] × 2
#   合并做 1 次 GEMM: [12,4096]@[4096,6144]
#   → GPU GEMM 效率随矩阵增大而提升（更好的 Tensor Core 利用率）


q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

# qkv [12, 6144] 在最后一维切分：
#   q = qkv[:, 0:4096]      → shape [12, 4096]  = [12, 32 heads × 128 dim]
#   k = qkv[:, 4096:5120]   → shape [12, 1024]  = [12, 8 heads × 128 dim]
#   v = qkv[:, 5120:6144]   → shape [12, 1024]  = [12, 8 heads × 128 dim]

# GQA 体现：Q 有 32 个 head，K/V 只有 8 个 head
# 每 4 个 Q head 共享 1 个 K/V head

# RoPE 旋转位置编码

q, k = self.rotary_emb(positions, q, k)

# positions = [10, 25, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0]
# RoPE 对 q 和 k 的每一行按对应 position 施加旋转变换
# q.shape 不变 = [12, 4096]
# k.shape 不变 = [12, 1024]
# 但数值已编码了位置信息

attn_output = self.attn(q, k, v)

#FlashAttention kernel 需要 3D [num_tokens, num_heads, head_dim] 格式
# q: [12, 4096] → view → [12, 32, 128]     (32 Q heads, 每头 128 dim)
# k: [12, 1024] → view → [12,  8, 128]     (8 KV heads, 每头 128 dim)
# v: [12, 1024] → view → [12,  8, 128]
# output: [12, 4096] → view → [12, 32, 128] 

# FlashAttentionImpl.forward() 接收的张量已经是 3D：
#   query:  [12, 32, 128]    ← FlashAttention 期望的格式
#   key:    [12,  8, 128]
#   value:  [12,  8, 128]
#   output: [12, 32, 128]    ← kernel 直接写入这里

attn_output.shape = [12, 4096]  ← 回到 2D，准备进 o_proj

output, _ = self.o_proj(attn_output)

# RowParallelLinear:
#   W_o shape: [4096, 4096]  (num_heads*head_dim → hidden_size)
#   output = attn_output @ W_o^T = [12, 4096] @ [4096, 4096]^T → [12, 4096]
#
# output.shape = [12, 4096]  ← 与输入 hidden_states 形状相同

```

```
hidden_states  [12, 4096]
     │
     ▼ qkv_proj (一次 GEMM)
qkv            [12, 6144]
     │
     ▼ split
q [12, 4096]   k [12, 1024]   v [12, 1024]
     │              │               │
     ▼ RoPE         ▼ RoPE          │
q [12, 4096]   k [12, 1024]   v [12, 1024]
     │              │               │
     ▼ view         ▼ view          ▼ view
q [12,32,128]  k [12,8,128]   v [12,8,128]
     │              │               │
     └──────────────┴───────────────┘
                    │
                    ▼ FlashAttention kernel
              attn_output [12, 4096]
                    │
                    ▼ o_proj
              output [12, 4096]
                    │
                    ▼ 回到 LlamaDecoderLayer → residual add → MLP → 下一层
融合多个操作减少显存带宽,本质 kernel Fusion。 
```
unified_attention_with_output 分发
```python
从全局 ForwardContext 获取该层的 metadata 和 kv_cache
    attn_metadata, attn_layer, kv_cache = get_attention_context(layer_name)

[2, N, block_size, num_kv_heads, head_dim]

perfix 逻辑。

flash_metadata = FlashAttentionMetadata(
    num_actual_tokens = 3,
    max_query_len     = 1,
    query_start_loc   = tensor([0, 1, 2, 3]),
    max_seq_len       = 158,
    seq_lens          = tensor([148, 158, 143]),
    block_table       = tensor([...]),     # 含公共前缀的 block + 各自后缀 block
    slot_mapping      = tensor([...]),

    use_cascade       = True,              # ← 开启 cascade
    common_prefix_len = 128,
    cu_prefix_query_lens = tensor([0, 3]), # 所有 3 个 query 一起处理前缀
    prefix_kv_lens    = tensor([128]),     # 前缀 128 个 KV
    suffix_kv_lens    = tensor([20, 30, 15]),  # 148-128, 158-128, 143-128
    max_num_splits     = 0,
    causal             = True,
)

```
写入 reshape_and_cache_flash， FlashAttention 不可用时 fallback triton_reshape_and_cache_flash。 

```
req_A: [system(128 tokens) + user_A(20 tokens)] → seq_len=148, 本步 1 decode
req_B: [system(128 tokens) + user_B(30 tokens)] → seq_len=158, 本步 1 decode
req_C: [system(128 tokens) + user_C(15 tokens)] → seq_len=143, 本步 1 decode

公共前缀: 128 tokens = 8 blocks（block_size=16）

# Cascade 两阶段：先处理公共前缀，再处理后缀
cascade_attention(
    output, query, key_cache, value_cache,
    cu_prefix_query_lens, prefix_kv_lens,   # 第 1 阶段
    suffix_kv_lens,                          # 第 2 阶段
    ...
)

Stage 1（公共前缀）:
  [query_A, query_B, query_C] 一起 attend → KV[0..127]
  公共前缀的 128 个 KV 只从 HBM 读 1 次
  3 个 query 在同一个 kernel call 中处理

Stage 2（各自后缀）:
  query_A attend → KV[128..147]   读 20 个 KV
  query_B attend → KV[128..157]   读 30 个 KV
  query_C attend → KV[128..142]   读 15 个 KV

→ 总 KV 读取 = 128（前缀1次） + 20 + 30 + 15 = 193 个 KV

对比：449 → 193，HBM 读取量减少 57%。cache line 利用率更高。 

```

## FlashAttention和FlashInfer

```

flash_attn_varlen_func
BatchPrefillWithPagedKVCacheWrapper

FlashInfer prefill和decode分开。 
prefill_output = self.prefill_wrapper.run(
                q_prefill, kv_cache
            )
decode_output = self.decode_wrapper.run(
    q_decode, kv_cache
)  

环境变量 VLLM_ATTENTION_BACKEND=FLASHINFER:
  → FlashInferBackend (强制使用 FlashInfer)
```

## PagedAttention v2
```python
def paged_attention_v2(
    out:          torch.Tensor,        # [num_seqs, num_heads, head_size]           最终输出
    exp_sums:     torch.Tensor,        # [num_seqs, num_heads, max_num_partitions]  每 partition 的 exp 之和
    max_logits:   torch.Tensor,        # [num_seqs, num_heads, max_num_partitions]  每 partition 的最大 score
    tmp_out:      torch.Tensor,        # [num_seqs, num_heads, max_num_partitions, head_size]  每 partition 的局部输出
    query:        torch.Tensor,        # [num_seqs, num_heads, head_size]
    key_cache:    torch.Tensor,        # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache:  torch.Tensor,        # [num_blocks, num_kv_heads, head_size, block_size]
    num_kv_heads: int,
    scale:        float,               # 1 / sqrt(head_dim)
    block_tables: torch.Tensor,        # [num_seqs, max_num_blocks_per_seq]
    seq_lens:     torch.Tensor,        # [num_seqs]
    block_size:   int,
    max_seq_len:  int,
    ...
) -> None

v1 串行处理head的全部 kv blocks。 

v2 将 KV 序列分区（Partition）并行。 假设 KV 序列长 2048，分成 4 个 partition（每段 512）：

  Partition 0: KV[0:512]
  Partition 1: KV[512:1024]
  Partition 2: KV[1024:1536]
  Partition 3: KV[1536:2048]

每个 partition p 独立计算局部 attention

  q shape:       [head_size]              ← 一个 head 的 query（如 [128]）
  K_p shape:     [PARTITION_SIZE, head_size] = [512, 128]  ← 该 partition 内的 KV
  V_p shape:     [PARTITION_SIZE, head_size] = [512, 128]
  out_p    → [head_size]=[128] ← 局部
每个 partition 保存 (m_p, exp_sum_p, out_p) 三元组，reduce kernel 重建全局 softmax 结果。 

paged_attention_v2_kernel
    grid = (num_heads, num_seqs, max_num_partitions)
paged_attention_v2_reduce_kernel
    grid = (num_heads, num_seqs) 

v1:  grid = (32 heads, 1 seq)           = 32 个 thread blocks
     每个 block 串行处理 8192/16 = 512 个 KV blocks → 慢

v2:  grid = (32 heads, 1 seq, 16 parts) = 32 × 16 = 512 个 thread blocks
     每个 block 只处理 512 个 token → 16x 并行度提升！
     + reduce grid = (32, 1) = 32 个 blocks（很快）

本质是并行。 FA2 GPU 利用率更高。 
```

## KV 布局

### NHD — FlashAttention 默认

```
key_cache shape: [num_blocks, block_size, num_heads, head_size]
               = [num_blocks,    16,         8,        128   ]
  block 0:
    token 0: head0[128 floats] head1[128 floats] ... head7[128 floats]  ← 1024 元素
    token 1: head0[128 floats] head1[128 floats] ... head7[128 floats]
    ...
    token 15: head0[128 floats] ... head7[128 floats]

物理地址示意（→ 代表内存地址递增）：
  → [blk0][tok0][h0][d0 d1 d2 ... d127] [blk0][tok0][h1][d0 ... d127] ...
  → [blk0][tok1][h0][d0 d1 d2 ... d127] ...

  ✓ 同一 token 的同一 head 的 128 维向量在内存中连续
  ✓ 同一 token 的所有 head 也连续（方便一次拷贝整个 token 的 KV）
  ✗ 不同 token 的同一 head 不连续

FlashAttention 用 block_table 做页级寻址，每次加载一整个 block 的 KV [i, block_size, num_heads, head_size]。 

```
### HND — FlashInfer 偏好

```
key_cache shape: [num_blocks, num_heads, block_size, head_size]
               = [num_blocks,    8,        16,        128   ]
block 0:
    head 0: token0[128] token1[128] ... token15[128]  ← 一个 head 的所有 token 连续
    head 1: token0[128] token1[128] ... token15[128]
    ...
    head 7: token0[128] ... token15[128]

物理地址示意：
  → [blk0][h0][tok0][d0..d127] [blk0][h0][tok1][d0..d127] ... [blk0][h0][tok15][d0..d127]
  → [blk0][h1][tok0][d0..d127] ...

特点：
  ✓ 同一 head 的不同 token 的向量在内存中连续
  ✓ GQA 场景下，只需要读少数 KV head，这些 head 的数据紧凑在一起
  ✗ 同一 token 的不同 head 不连续

GQA 32:8 意味着 32 个 Q head 分成 8 组，每组 4 个 Q head 共享 1 个 KV head。

假设 GPU 某个 SM 当前只需要处理 Q head 0~3（一组），它只需要读 KV head 0。 本质按head读取。 


  FlashInfer 的 decode kernel 按 head 划分工作，一次处理一个 head 的全部 token → 连续内存访问。

HND 的 reshape_and_cache_flash_kernel 需要走 per-head 循环，写会费事点。 
```
### PagedAttention（x 维度重排）
```
key_cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
               = [num_blocks,    8,        128/8,       16,       8 ]
               = [num_blocks,    8,         16,          16,       8 ]
  其中 x = 16 / sizeof(fp16) = 8

PagedAttention kernel 的工作方式：
  一个 warp (32 threads) 处理一个 KV block（16 个 token）
  → THREAD_GROUP_SIZE = WARP_SIZE / BLOCK_SIZE = 32/16 = 2
  → 每 2 个线程组成一个 thread group，负责一个 token 的 dot product
  → 每个 thread group 一次读取 16 bytes = 8 个 fp16 = x 个元素

线程读取 K 的模式：
  thread_group_0 (thread 0,1): 负责 token 0 的 q·k dot product
    thread 0 读 k[token0][dim 0:8],  k[token0][dim 16:24], ...
    thread 1 读 k[token0][dim 8:16], k[token0][dim 24:32], ...

  thread_group_1 (thread 2,3): 负责 token 1
    thread 2 读 k[token1][dim 0:8],  k[token1][dim 16:24], ...
    thread 3 读 k[token1][dim 8:16], k[token1][dim 24:32], ...

  ...以此类推，thread_group_15 负责 token 15

    NHD 布局（token 维连续，每 token 128 dim 连续）：
  ┌─────────────────────────────────────────────┐
  │ tok0: d0 d1 d2 ... d127 │ tok1: d0 d1 ... │ ...
  └─────────────────────────────────────────────┘
  thread0 ↑                   thread2 ↑           ← 间隔 256 bytes

  PagedAttention 重排布局（chunk 内 token 维连续，每 chunk 8 dim）：
  ┌─────────────────────────────────────────────────────┐
  │ chunk0: t0[d0..7] t1[d0..7] t2[d0..7] ... t15[d0..7] │
  │ chunk1: t0[d8..15] t1[d8..15] ...                      │
  └─────────────────────────────────────────────────────┘
  thread0 ↑    thread2 ↑    thread4 ↑                  ← 间隔 16 bytes → coalesced！


  为了 相邻 thread 读相邻地址 → coalesced！  
  Coalesced Memory Access（合并内存访问）， HBM 一次事务 = 32 bytes 或 128 bytes。

```


# Forward 


3 个 decode 请求，pad 到 4（CUDA Graph），Llama-7B，FlashAttention 后端：

```
req_1: seq_len=11, 1 decode token
req_2: seq_len=26, 1 decode token
req_3: seq_len=6,  1 decode token
```

```python
common = CommonAttentionMetadata(
    query_start_loc = [0, 1, 2, 3, 4],    # [num_reqs+1=4]
    seq_lens        = [11, 26, 6],          # [num_reqs=3]
    num_reqs        = 3,
    num_actual_tokens = 4,                  # 含 1 pad
    max_query_len   = 1,                    # decode
    max_seq_len     = 26,
    block_table     = [[blk0,-1], [blk5,blk6], [blk2,-1]],
    slot_mapping    = [170, 121, 37, 0],    # 3 real + 1 pad
    causal          = True,
)
flash_metadata = FlashAttentionMetadata(
    num_actual_tokens = 4,
    max_query_len     = 1,
    query_start_loc   = [0, 1, 2, 3, 4],
    max_seq_len       = 26,
    seq_lens          = [11, 26, 6],
    block_table       = [[blk0,-1], [blk5,blk6], [blk2,-1]],
    slot_mapping      = [170, 121, 37, 0],
    causal            = True,
    use_cascade       = False,              # 无公共前缀
    scheduler_metadata = None,              # FA2 不需要
)

forward_context.attn_metadata = {
    "model.layers.0.self_attn": flash_metadata,  # 32 层共用同一对象
    ...
    "model.layers.31.self_attn": flash_metadata,
}

模型 forward 中每层 Attention
# 以 layer 0 为例：
# LlamaAttention.forward():
#   q = Wq @ hidden  →  [4, 32, 128]  (4 tokens, 32 heads, 128 dim)
#   k = Wk @ hidden  →  [4, 32, 128]
#   v = Wv @ hidden  →  [4, 32, 128]
#   apply_rotary_pos_emb(q, k, positions=[10,25,5,0])

# Attention.forward() → unified_attention_with_output()
#   → get_attention_context("model.layers.0.self_attn")
#   → FlashAttentionImpl.forward(layer, q, k, v, kv_cache, flash_metadata, output)

# reshape_and_cache（写入新 K/V）
triton_reshape_and_cache_flash(
    key=k,               # [4, 32, 128]
    value=v,             # [4, 32, 128]
    key_cache=key_cache, # [1024, 16, 32, 128]
    value_cache=...,
    slot_mapping=[170, 121, 37, 0],
)
# → key_cache[10][10] ← k[0]   (slot 170 = 10*16+10)
# → key_cache[7][9]   ← k[1]   (slot 121 = 7*16+9)
# → key_cache[2][5]   ← k[2]   (slot 37 = 2*16+5)
# → key_cache[0][0]   ← k[3]   (slot 0, pad, 无效写入)

flash_attn_varlen_func(
    q            = query[:4],              # [4, 32, 128]
    k            = key_cache,              # Paged
    v            = value_cache,            # Paged
    out          = output[:4],             # [4, 32, 128]
    cu_seqlens_q = [0, 1, 2, 3, 4],       # 每请求 1 个 query
    seqused_k    = [11, 26, 6],            # 每请求读取的 KV 数
    max_seqlen_q = 1,
    max_seqlen_k = 26,
    block_table  = [[blk0,-1],[blk5,blk6],[blk2,-1]],
    causal       = True,
)
# → output[:4] shape [4, 32, 128] → view → [4, 4096]

# o_proj
# output = Wo @ attn_output  → [4, 4096]
```