---
title: Fish AR Omni
date: 2026-06-07 22:00:00 +0800
categories: [Fish, Realtime, VoiceAgent]
tags: [fish, realtime, voice, multimodal, omni]
pin: true
---
## Fish流水线

```
 Fish Audio S2-Pro  (Dual-AR TTS, 采样率 44.1kHz, 帧率 ~21Hz, 10 codebooks)

 "请用开心的语气说：今天天气真好 [laugh]"   (+ 可选参考音频做声音克隆)
        │
        ▼
 ┌─────────────────────────┐
 │ Stage1  Preprocessing   │  CPU
 │  - 文本 → Qwen3 chat 模板 │ 
 │  - 参考音频 → DAC.encode  │  ref audio → VQ codes，prepend 进 system prompt
 └──────────┬──────────────┘
            │  input_ids + vq_parts + vq_mask
            ▼
 ╔═════════════════════════════════════════════════════╗
 ║ Stage2  Dual-AR 生成引擎  GPU        ║
 ║                                                       ║
 ║   ┌───────────────────────────────────────────┐     ║
 ║   │ Slow AR (Temporal Semantic Backbone)       │     ║
 ║   │  Qwen3 decoder-only, ~4B                    │     ║
 ║   │  沿【时间轴】每步预测 1 个语义 token          │     ║
 ║   │  RMSNorm + SwiGLU + GQA + RoPE + QK-Norm    │     ║
 ║   └───────────────┬───────────────────────────┘     ║
 ║       hidden state │ (每个时间步 t 输出一个 h_t)        ║
 ║                    ▼                                   ║
 ║   ┌───────────────────────────────────────────┐     ║
 ║   │ Fast AR (Depth-wise Acoustic Decoder)      │     ║
 ║   │  4 层 Transformer, ~400M, 独立权重/embedding │     ║
 ║   │  沿【深度轴】自回归生成 9 个残差 codebook      │     ║
 ║   │  循环 9 步: h_t → cb1 → cb2 → ... → cb9      │     ║
 ║   └───────────────┬───────────────────────────┘     ║
 ║                    │  1 帧 = 1 语义 + 9 残差 = 10 码    ║
 ║   下一时间步: 10 个码 sum→embedding 注回 Slow AR 输入   ║
 ╚════════════════════╤════════════════════════════════╝
                      │  累积的 codebook indices (T, 10)
                      ▼
 ┌─────────────────────────┐
 │ Stage3  DAC Vocoder     │  GPU
 │  ModifiedDAC 解码         │  (EVA-GAN 风格生成器)
 │  codes → 44.1kHz 波形     │  支持流式分块输出
 └──────────┬──────────────┘
            ▼
        最终音频 (wav)
```
差异点：
```
标准 GPT:
  hidden state h_t ──▶ LM Head ──▶ 文本 token logits ──▶ 采样下一个词
                        （唯一出口）

Fish S2 Slow AR:
  hidden state h_t ──▶ LM Head ──▶ 语义 token logits ──▶ 采样 sem_t
                   │
                   └──▶ project_in ──▶ Fast AR（补全 9 层残差码）
                        （第二个出口，这是 GPT 没有的）
```
 
**性能（单卡 H200, batch=1）**：RTF ≈ 0.34，63.3 tok/s，TTFT ≈ 18ms，Time-to-First-Audio ≈ 140ms。
Backbone 标准 Qwen3 decoder-only（等价 LLM），自回归预测离散 audio token。