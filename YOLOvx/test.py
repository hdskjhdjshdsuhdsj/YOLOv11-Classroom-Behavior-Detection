from ultralytics import YOLO


def test_model():
    """在测试集上评估模型"""
    print("🎯 在测试集上进行最终评估...")

    # 加载新训练的最佳模型（7类课堂行为，50轮）
    model = YOLO('./runs/detect/class_50ep/weights/best.pt')

    # 在测试集上评估模型
    model.val(
        data='../class/data.yaml',  # 7类课堂行为数据集
        split='test',               # 使用测试集
        save_json=True,             # 保存JSON结果
    )

    print("✅ 最终测试完成!")


def main():
    # 测试模型
    test_model()


if __name__ == "__main__":
    main()
