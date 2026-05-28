---
title: OpenAI Realtime
date: 2026-05-24 22:00:00 +0800
categories: [OpenAI, Realtime, VoiceAgent]
tags: [openai, realtime, voice, multimodal]
pin: true
---

## 并行流协议
```
Realtime 是 OpenAI 端到端语音模型——不再走 STT → LLM → TTS 三段拼接，而是一个模型同时吃 PCM 出 PCM，保留了语气、笑声、犹豫、停顿等"文字丢掉的信息"，端到端首音延迟 降到 ~300ms。
2024-10-01  gpt-4o-realtime-preview-2024-10-01
2025-08-28  gpt-realtime（GA
     · 加 image 输入（多模态）
    · semantic_vad 作为推荐默认
2025-10-06  gpt-realtime-mini-2025-10-06           ── 比 GA 主力便宜 ~70%
最新 gpt-realtime-2（SDK DEFAULT_REALTIME_MODEL）
协议的设计很不错～，本质上是一个持续存在的、多流并行、事件驱动的会话协议（streaming event protocol）。 
```
### 正常turn 
```
Client (App)                                       OpenAI Realtime Server
   │                                                 │
   │ 每 40ms 推一帧 PCM16 base64                       │
   │── { type:"input_audio_buffer.append",           │   ④ 上行 PCM
   │     audio:"<base64 ~40ms>" } ──────────────────▶│
   │── ... append × N ──────────────────────────────▶│
   │                                                 │
   │◀── { type:"input_audio_buffer.speech_started",  │   ⑤ server VAD 检测到开口
   │      audio_start_ms:4044 } ◀────────────────────│      → raw_model_event(raw_server_event)
   │◀── { type:"input_audio_buffer.speech_stopped",  │   ⑥ VAD 检测到说完
   │      audio_end_ms:5680 } ◀──────────────────────│
   │◀── { type:"input_audio_buffer.committed",       │   ⑦ server 把刚才那段 PCM 提交为一个 item
   │      item_id:"item_..." } ◀─────────────────────│
   │◀── { type:"conversation.item.added",            │   ⑧ user item 入 history
   │      item:{ role:"user", type:"message",        │      → raw_model_event(item_updated)
   │             content:[{ type:"input_audio" }] }} │    
   │◀── { type:"conversation.item.done", item } ◀────│     
   │                                                 │
   │◀── { type:"response.created",                   │   ⑨ ★ 一个 turn 正式开始
   │      response:{ id:"resp_DkPw88XR...",          │      → raw_model_event(turn_started)
   │                 status:"in_progress" }} ◀───────│      → RealtimeSessionEvent.agent_start
   │                                                 │
   │◀── { type:"conversation.item.input_audio_       │   ⑩ ★ ASR 转写流：用户那一句
   │       transcription.delta", delta:"嗨" } ◀───────│      （独立于 ⑫，描述 user 而不是 assistant）
   │◀── ... transcription.delta × N ◀────────────────│
   │◀── { type:"conversation.item.input_audio_       │
   │       transcription.completed",                 │      → raw_model_event(
   │       transcript:"<你刚说的话>" } ◀───────────────│           input_audio_transcription_completed)
   │                                                 │
   │◀── { type:"response.output_item.added",         │   ⑪ assistant message item 的"骨架"
   │      item:{ type:"message", id:"msg_..." }} ◀───│      → raw_model_event(item_updated)
   │◀── { type:"conversation.item.added", item } ◀───│      assistant item 进 history
   │◀── { type:"response.content_part.added",        │
   │      part:{ type:"audio" } } ◀──────────────────│
   │                                                 │
   │◀── { type:"response.output_audio_transcript.    │   ⑫ ★ 文字流（与 ⑬ 音频流并行）
   │       delta", delta:"嗨", item_id,              │      → raw_model_event(transcript_delta)
   │       content_index, response_id } ◀────────────│
   │◀── { type:"response.output_audio.delta",        │   ⑬ ★ PCM16 base64 增量
   │      delta:"<base64 PCM16>" } ◀─────────────────│      → RealtimeSessionEvent.audio
   │◀── ... transcript.delta + audio.delta 交错 N 帧 ◀│      （喂播放器；同时累计 ms 进 tracker）
   │                                                 │
   │◀── { type:"response.output_audio.done" } ◀──────│   ⑭ assistant 音频流完
   │                                                 │      → RealtimeSessionEvent.audio_end
   │◀── { type:"response.output_audio_transcript.    │
   │       done" } ◀─────────────────────────────────│
   │◀── { type:"response.content_part.done" } ◀──────│
   │◀── { type:"conversation.item.done" } ◀──────────│
   │◀── { type:"response.output_item.done" } ◀───────│
   │◀── { type:"response.done",                      │   ⑮ ★ turn 收尾（usage 在这里）
   │      response:{ id:"resp_DkPw88XR...",          │      → raw_model_event(turn_ended)
   │                 status:"completed",             │      → RealtimeSessionEvent.agent_end
   │                 usage:{ ... }}} ◀───────────────│
```

### Barge-in
```
Client (App)                                       OpenAI Realtime Server
   │                                                 │
   │ ... 仍在听 resp_DkPwMRtB... 的 output_audio.delta ...
   │                                                 │
   │ 用户开口；麦克风照常推上行 PCM                       │
   │── input_audio_buffer.append × N ───────────────▶│
   │                                                 │
   │◀── { type:"input_audio_buffer.speech_started",  │   ⓪ ★ 打断信号
   │      audio_start_ms:23788, item_id } ◀──────────│      
   │                                                 │       
   │     [App] _audio_state_tracker.on_interrupted() │
   │           emit RealtimeModelAudioInterrupted    │      → RealtimeSessionEvent.audio_interrupted
   │           player.stop() / 清本地缓冲              │      → 前端 UI 立刻停播 + 视觉反馈
   │                                                 │
   │── { type:"conversation.item.truncate",          │   App 主动告诉 server "我只播到 N ms"
   │     item_id:"<被打断那条 assistant>",            │     audio_end_ms 由谁算：
   │     content_index:0,                            │       · PlaybackTracker → 用前端
   │     audio_end_ms:<实际播放进度ms> } ─────────────▶│         实际播放进度
   │                                                 │       · ModelAudioTracker 的
   │                                                 │         发送字节累计兜底
   │                                                 │
   │ ─── server 把当前 response 截短并收尾 ───           │
   │◀── { type:"response.output_audio.done" } ◀──────│   ① 老 turn 的音频段被服务端 cut 短
   │◀── { type:"response.output_audio_transcript.    │
   │       done" } ◀─────────────────────────────────│
   │◀── { type:"response.content_part.done" } ◀──────│
   │◀── { type:"conversation.item.done" } ◀──────────│
   │◀── { type:"response.output_item.done" } ◀───────│
   │◀── { type:"response.done",                      │   ② ★ 老 response 收尾
   │      response:{ id:"resp_DkPwMRtB...",          │     ⚠ 状态仍然是 "completed"，
   │                 status:"completed" }} ◀─────────│       
   │                                                 │      → raw_model_event(turn_ended)
   │                                                 │      → RealtimeSessionEvent.agent_end
   │                                                 │
   │◀── { type:"conversation.item.truncated",        │   ③ ★ server ack 截断；history 里
   │      item_id, content_index,                    │      那条 assistant audio item 永久变短
   │      audio_end_ms } ◀──────────────────────────│      
   │◀── { type:"conversation.item.retrieved",        │
   │      item:{ ... } } ◀───────────────────────────│
   │                                                 │
   │◀── { type:"input_audio_buffer.speech_stopped",  │   ④ 用户说完
   │      audio_end_ms:25960 } ◀─────────────────────│
   │◀── { type:"input_audio_buffer.committed" } ◀────│
   │◀── { type:"conversation.item.added",            │   ⑤ user 新一句进 history
   │      item:{ role:"user", ... }} ◀───────────────│
   │◀── { type:"conversation.item.done" } ◀──────────│
   │                                                 │
   │◀── { type:"response.created",                   │   ⑥ 新 turn 开始
   │      response:{ id:"resp_DkPwWuuR...",          │      → raw_model_event(turn_started)
   │                 status:"in_progress" }} ◀───────│   ... 回到 ##### 1 的 ⑫⑬⑭⑮ ...
```


## Agent Normal turn 

```
User Mic     App        RealtimeSession      OpenAIRealtimeWS         OpenAI Server
  │           │              │                     │                       │
  │  PCM16    │              │                     │                       │
  ├──────────▶│ send_audio() │                     │                       │
  │           ├─────────────▶│ send_event(         │                       │
  │           │              │   SendAudio)        │                       │
  │           │              ├────────────────────▶│  input_audio_buffer.  │
  │           │              │                     │  append ─────────────▶│
  │           │              │                     │                       │
  │           │              │                     │◀── speech_started ────│ (server VAD)
  │           │              │◀── raw_model_event ─┤   (raw_server_event)  │
  │           │              │                     │◀── speech_stopped ────│
  │           │              │                     │◀── item.created (user)│
  │           │              │◀── item_updated ────┤                       │
  │           │   history_added                    │                       │
  │           │              │                     │◀── response.created ──│
  │           │              │◀── turn_started ────┤                       │
  │           │   agent_start│                     │                       │
  │           │              │                     │◀── audio_transcript.  │
  │           │              │                     │    delta ×N ─────────│
  │           │              │◀── transcript_delta─┤ (累积进 _item_transcripts，
  │           │              │                     │  到阈值跑 guardrail)  │
  │           │              │                     │◀── audio.delta ×N ────│
  │           │              │◀── audio  ──────────┤ (ModelAudioTracker++) │
  │           │   audio      │                     │                       │
  │           │◀ event ──────│                     │                       │
  │  PCM ◀────│ player.feed  │                     │                       │
  │           │ playback_tracker.on_play_bytes ────────────────────────────▶ (本地)
  │           │              │                     │◀── audio.done ────────│
  │           │              │◀── audio_done ──────┤                       │
  │           │   audio_end  │                     │                       │
  │           │              │                     │◀── response.done ─────│
  │           │              │◀── turn_ended ──────┤                       │
  │           │   agent_end  │                     │                       │
```
```
handoff 走同一个 function_call 通道：tool_name == handoff_tool_name
→ Session 切换 _current_agent，emit handoff event，再 session.update 把新 agent的 tools / instructions / voice 推给服务端。
```

## Agent Barge-in
```
User Mic     App        RealtimeSession      OpenAIRealtimeWS         OpenAI Server
  │           │              │                     │                       │
  │ ... 仍在听 resp_X 的 audio.delta（已播 N 毫秒）...                          │
  │           │              │                     │                       │
  │  PCM16    │              │                     │                       │
  ├──────────▶│ send_audio() │                     │                       │
  │           ├─────────────▶│ send_event(         │                       │
  │           │              │   SendAudio)        │                       │
  │           │              ├────────────────────▶│  input_audio_buffer.  │
  │           │              │                     │  append ─────────────▶│
  │           │              │                     │                       │
  │           │              │                     │◀── speech_started ────│ ① server VAD 触发
  │           │              │                     │   (audio_start_ms)    │   
  │           │              │                     │                       │     
  │           │              │                     │                       │
  │           │              │                     │  _audio_state_tracker │
  │           │              │                     │  .on_interrupted()    │
  │           │              │◀── audio_interrupted┤  emit RealtimeModel   │
  │           │              │   (RealtimeModel    │   AudioInterruptedEvt │
  │           │              │    AudioInterrupted)│                       │
  │           │   audio_interrupted                │                       │
  │           │◀ event ──────│                     │                       │
  │  停播 ◀──┤ player.stop()│                     │                       │
  │           │   清空本地    │                     │                       │
  │           │   缓冲        │                     │                       │
  │           │              │                     │                       │
  │           │              │  audio_end_ms ←     │                       │
  │           │              │   playback_tracker  │                       │
  │           │              │   .get_state()      │                       │
  │           │              │   (没装 tracker →    │                       │
  │           │              │    ModelAudioTracker│                       │
  │           │              │    发送字节兜底)      │                       │
  │           │              │                     │                       │
  │           │              │  send_event(        │                       │
  │           │              │   SendInterrupt)    │                       │
  │           │              ├────────────────────▶│  conversation.item.   │ ② App 主动告诉 server
  │           │              │                     │  truncate(item_id,    │   "我只播到 N ms"
  │           │              │                     │   content_index,      │
  │           │              │                     │   audio_end_ms) ────▶│
  │           │              │                     │                       │
  │           │              │                     │◀── output_audio.done ─│ ③ server 截短当前 audio
  │           │              │◀── audio_done ──────┤                       │
  │           │   audio_end  │                     │                       │
  │           │              │                     │◀── output_audio_     │
  │           │              │                     │    transcript.done ───│
  │           │              │                     │◀── content_part.done ─│
  │           │              │                     │◀── conversation.item. │
  │           │              │                     │    done ──────────────│
  │           │              │                     │◀── response.output_   │
  │           │              │                     │    item.done ─────────│
  │           │              │                     │◀── response.done ─────│ ④ 老 turn 收尾
  │           │              │                     │   (status="completed",│   ⚠completed
  │           │              │                     │    不是 cancelled)     │     
  │           │              │◀── turn_ended ──────┤                       │
  │           │   agent_end  │                     │                       │
  │           │              │                     │                       │
  │           │              │                     │◀── conversation.item. │ ⑤ server ack 截断；
  │           │              │                     │    truncated ─────────│   history 中那条 assistant
  │           │              │◀── item_updated ────┤                       │   audio 变短
  │           │   history_updated                  │                       │   （影响下一 turn 上下文）
  │           │              │                     │                       │
  │  ... 用户说完，回到 ##### 2：speech_stopped → committed → user item.added           │
  │       → response.created（新 turn）→ transcript_delta / audio.delta ...           │
```