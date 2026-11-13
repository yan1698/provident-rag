"""Print the 3-minute MVP demo path. Does not call the generator LLM."""

from __future__ import annotations

BASE = "http://localhost:8000"

print(
    f"""Provident 3 分钟 Demo
====================
1. 打开溯源页  {BASE}/ui
2. 上传自己的文档，完成向量化后，根据文档内容自行提问
3. 点答案里的 [S1]，右侧证据高亮；看 Verifier 逐句 supported / unsupported
4. 打开评测页  {BASE}/ui/eval   （闸门 PASS/FAIL，不调生成 LLM）
5. 可选 Agent {BASE}/ui/agent  观察 retrieve → answer → verify 审计日志
6. 指标       {BASE}/metrics

API 文档 {BASE}/docs
"""
)
