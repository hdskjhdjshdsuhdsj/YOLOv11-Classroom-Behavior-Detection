"""
main.py — 命令行入口

用法：
  python main.py --mock                      # 用假数据跑通链路，不联网不要权重
  python main.py --image path/to/photo.jpg   # 真检测 + LLM 分析
  python main.py --image xxx.jpg --no-llm    # 只检测，不调用 AI
"""
import argparse
import json
from pathlib import Path

from detector import Detector, mock_detections
from llm_client import LLMClient


ROOT = Path(__file__).resolve().parent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="YOLO 课堂行为检测 + LLM 智能分析 Agent")
    parser.add_argument("--config", default=str(ROOT / "config.json"), help="配置文件路径")
    parser.add_argument("--image", help="要分析的图片路径")
    parser.add_argument("--mock", action="store_true", help="用内置假数据跑通链路（无需权重/联网）")
    parser.add_argument("--no-llm", action="store_true", help="只输出检测结果，不调用 LLM")
    parser.add_argument("--save", action="store_true", help="把报告保存到 reports/ 目录")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. 拿到检测结果
    if args.mock:
        print("[*] 使用模拟检测结果（mock 模式）")
        detections = mock_detections()
    elif args.image:
        print(f"[*] 检测图片: {args.image}")
        detections = Detector(cfg["detector"]).detect(args.image)
    else:
        parser.error("请指定 --image 或 --mock 之一")

    print(f"[*] 检测到 {len(detections)} 个目标")
    for d in detections:
        print(f"    - {d['class']:<16} 置信度={d['confidence']:.2f}  bbox={d['bbox']}")

    if args.no_llm:
        print("\n[--no-llm] 跳过 AI 分析。")
        return

    # 2. 交给 LLM 分析
    focus = cfg.get("analysis", {}).get("focus", [])
    client = LLMClient(cfg["llm"])
    print(f"\n[*] LLM provider={client.provider}, model={client.model}")
    try:
        report = client.analyze(detections, focus)
    except Exception as exc:
        # 把 openai 的异常转成可读提示，避免用户面对一长串堆栈
        name = type(exc).__name__
        if "Authentication" in name or "401" in str(exc):
            print("\n[失败] DeepSeek 密钥无效或已过期。")
            print("  请检查 agent/api_key.txt 里的 key 是否正确。")
            print("  申请地址: https://platform.deepseek.com/api_keys")
        elif "Connection" in name or "Timeout" in name or "APIConnection" in name:
            print("\n[失败] 无法连接 DeepSeek，请检查网络或代理设置。")
        else:
            print(f"\n[失败] AI 分析出错: {name}: {exc}")
        raise SystemExit(1)

    # 3. 输出报告
    print("\n" + "=" * 50)
    print(report)
    print("=" * 50)

    if args.save:
        out_dir = ROOT / "reports"
        out_dir.mkdir(exist_ok=True)
        from datetime import datetime
        out_file = out_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        out_file.write_text(report, encoding="utf-8")
        print(f"\n[*] 报告已保存: {out_file}")


if __name__ == "__main__":
    main()
