from ultralytics import YOLO
import os

def resume_training():
    """恢复训练"""
    # 查找最新的训练权重(自行替换)
    weights_dir = r"E:\PartCounterSystem\ai_models\data\runs\train\weights"
    last_pt = os.path.join(weights_dir, "last.pt")

    if not os.path.exists(last_pt):
        print("❌ 未找到上次训练的权重文件")
        return

    print(f"恢复训练: {last_pt}")

    model = YOLO(last_pt)

    model.train(
        resume=True,  # 关键参数：恢复训练
        epochs=120,  # 总epochs数
        patience=15
    )

    print("恢复训练完成!")


if __name__ == "__main__":
    resume_training()