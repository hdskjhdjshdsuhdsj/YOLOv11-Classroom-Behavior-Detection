from ultralytics import YOLO


def main():
    # 论文模型：YOLOv11m（-medium，精度与速度的平衡）
    model = YOLO('yolo11m.pt')

    # 训练配置：课堂行为数据集（7类），50轮
    model.train(
        data='../class/data.yaml',   # 课堂行为数据集：sleep/raise_head/using_computer/stand/using_phone/raise_hand/read
        epochs=50,                    # 训练轮数
        imgsz=640,                    # 图像尺寸
        batch=8,                      # 批次大小（RTX 3060 6GB）
        lr0=0.01,                     # 初始学习率
        workers=1,
        patience=0,                   # 禁用早停，跑满50轮
        name='class_50ep',            # 输出目录 runs/detect/class_50ep
        exist_ok=True,
    )

    print("✅ 训练完成!")


if __name__ == '__main__':
    main()
