"""
detector.py — YOLO 检测层

职责：输入一张图片路径，输出结构化的行为检测结果。
依赖 ultralytics（YOLO 官方库）。
"""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


class Detector:
    def __init__(self, cfg: dict):
        self.conf = cfg.get("conf", 0.25)
        self.iou = cfg.get("iou", 0.45)
        self.imgsz = cfg.get("imgsz", 640)
        self.device = cfg.get("device", "") or None
        weights = cfg.get("weights", "weights/best.pt")

        # 权重路径相对本文件所在目录解析，避免受运行目录影响
        weight_path = Path(weights)
        if not weight_path.is_absolute():
            weight_path = Path(__file__).resolve().parent / weight_path
        if not weight_path.exists():
            raise FileNotFoundError(
                f"找不到 YOLO 权重: {weight_path}\n"
                f"请把训练好的 best.pt 放到 agent/weights/ 下，或修改 config.json 的 detector.weights"
            )
        self.model = YOLO(str(weight_path))

    def detect(self, image_path: str) -> list[dict]:
        """检测单张图片，返回行为列表。每个元素: {class, class_id, confidence, bbox}"""
        results = self.model.predict(
            source=image_path,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "class": results[0].names[cls_id],
                "class_id": cls_id,
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],  # [x1,y1,x2,y2]
            })
        return detections


def mock_detections() -> list[dict]:
    """无权重时的假数据，仅用于跑通后续 LLM 链路。"""
    return [
        {"class": "read", "class_id": 6, "confidence": 0.94, "bbox": [120, 80, 300, 420]},
        {"class": "using_computer", "class_id": 2, "confidence": 0.91, "bbox": [340, 90, 520, 430]},
        {"class": "using_phone", "class_id": 4, "confidence": 0.82, "bbox": [560, 150, 690, 380]},
        {"class": "sleep", "class_id": 0, "confidence": 0.78, "bbox": [700, 200, 880, 470]},
        {"class": "raise_hand", "class_id": 5, "confidence": 0.66, "bbox": [200, 60, 310, 260]},
    ]
