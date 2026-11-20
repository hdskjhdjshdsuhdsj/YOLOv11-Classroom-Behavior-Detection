"""
llm_client.py — LLM 分析层

职责：把检测结果组装成 prompt，发给大模型，拿回中文分析报告。
支持两种 provider：
  - openai_compatible：任何 OpenAI 兼容接口（官方 / 中转 / Ollama / vLLM 等）
  - mock：不联网，返回固定文本，用来跑通链路
"""
from __future__ import annotations

import json
import os
from collections import Counter


SYSTEM_PROMPT = """你是一名课堂行为分析助手。
我会给你一组目标检测模型输出的学生行为结果（类别、置信度、位置框），请用中文输出：
1. 行为统计：各类行为的数量与占比
2. 异常判断：指出玩手机、睡觉、站立、东张西望、趴桌子等异常行为及其严重程度
3. 课堂状态评价：整体纪律与专注度
4. 改进建议：针对异常情况给出具体建议
要求：只基于给定数据，不要臆造未检出的行为；表述简洁、有条理，用 Markdown 格式。"""


def build_prompt(detections: list[dict], focus: list[str]) -> str:
    """把检测结果转成结构化文本，交给 LLM。"""
    total = len(detections)
    counts = Counter(d["class"] for d in detections)
    stats = [
        {"行为": name, "数量": cnt, "占比": f"{cnt / total * 100:.1f}%"}
        for name, cnt in counts.most_common()
    ] if total else []

    payload = {
        "检测目标总数": total,
        "行为统计": stats,
        "检测明细": [
            {
                "行为": d["class"],
                "置信度": d["confidence"],
                "位置框_像素": d["bbox"],
            }
            for d in detections
        ],
        "重点关注的异常行为": focus,
    }
    return "以下是课堂行为检测结果（JSON）：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


class LLMClient:
    def __init__(self, cfg: dict):
        self.provider = cfg.get("provider", "mock")
        self.model = cfg.get("model", "gpt-4o-mini")
        self.temperature = cfg.get("temperature", 0.3)
        self.max_tokens = cfg.get("max_tokens", 1024)

        if self.provider == "openai_compatible":
            from openai import OpenAI  # 延迟导入：mock 模式无需安装 openai 库
            # 密钥只从环境变量读取，避免写进 config.json 被提交到公开仓库。
            # config 里的 api_key 仅作为本地临时兜底，正常应保持为空占位。
            api_key = os.environ.get("OPENAI_API_KEY") or cfg.get("api_key", "")
            base_url = os.environ.get("OPENAI_BASE_URL") or cfg.get(
                "base_url", "https://api.openai.com/v1"
            )
            if not api_key or api_key.startswith("sk-在这里"):
                raise ValueError(
                    "未检测到有效的 API key。请通过环境变量 OPENAI_API_KEY 提供，"
                    "不要把密钥写进 config.json（该文件会随仓库公开）。\n"
                    "  Windows CMD     : set OPENAI_API_KEY=你的key\n"
                    "  Windows PowerShell: $env:OPENAI_API_KEY=\"你的key\""
                )
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=cfg.get("timeout", 60),
            )
        elif self.provider == "mock":
            self.client = None
        else:
            raise ValueError(f"未知的 llm.provider: {self.provider}，可选 openai_compatible / mock")

    def analyze(self, detections: list[dict], focus: list[str]) -> str:
        prompt = build_prompt(detections, focus)
        if self.provider == "mock":
            return self._mock_response(detections, prompt)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def _mock_response(self, detections: list[dict], prompt: str) -> str:
        """打印实际会发出的 prompt，并返回模拟分析，方便检查组装是否正确。"""
        print("\n===== [MOCK] 实际发给 LLM 的 prompt 预览 =====")
        print("--- system ---")
        print(SYSTEM_PROMPT)
        print("--- user ---")
        print(prompt)
        print("===== prompt 预览结束 =====\n")
        return (
            "## 1. 行为统计\n"
            f"- 共检测到 {len(detections)} 个目标（此处为模拟数据）。\n\n"
            "## 2. 异常判断\n"
            "- 检测到 using_phone、sleep 各 1 例，属于明显异常行为。\n\n"
            "## 3. 课堂状态评价\n"
            "- 整体以正常听课为主，但存在个别学生注意力分散。\n\n"
            "## 4. 改进建议\n"
            "- 建议教师关注后排玩手机与睡觉的学生，适当提问或调整座位。\n"
        )
