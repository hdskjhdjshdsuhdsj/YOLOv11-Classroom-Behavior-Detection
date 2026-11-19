import cv2
import os
from ultralytics import YOLO

# ==================== 配置参数 ====================
# 论文模型：class 数据集 7 类课堂行为（50轮训练）
# sleep 睡觉 / raise_head 抬头 / using_computer 用电脑 / stand 站立
# using_phone 玩手机 / raise_hand 举手 / read 读书
MODEL_PATH = "./runs/detect/class_50ep/weights/best.pt"  # 模型权重路径（论文7类模型）
IMAGE_PATH = "test_image.jpg"                       # 默认图片
CONFIDENCE_THRESHOLD = 0.2                        # 🔥 实时建议 0.2~0.25
IMG_SIZE = 640                                      # 🔥 统一输入尺寸


# ==================== 图像预测 ====================
def predict_image(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return

    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功（图像预测）")

    results = model(
        image_path,
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    img = results[0].plot()
    cv2.imshow("图像检测结果", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================== 🚀 实时摄像头检测（重点） ====================
def predict_camera():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    model.fuse()  # 🔥 提速
    print("✅ 模型加载成功，开始实时摄像头检测")
    print("🎥 按 q 键退出")

    # 🔥 官方推荐实时模式
    for result in model.predict(
        source=0,                      # 摄像头
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMG_SIZE,
        stream=True,                   # 🔥 核心：实时流
        verbose=False
    ):
        frame = result.plot()
        cv2.imshow("实时检测", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ==================== 视频文件检测 ====================
def predict_video(video_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return

    model = YOLO(MODEL_PATH)
    model.fuse()
    print("✅ 模型加载成功，开始视频检测")
    print("🎥 按 q 键退出")

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        frame = results[0].plot()
        cv2.imshow("视频检测", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🎯 模型识别能力测试")
    print("=" * 30)
    print("1. 图像预测")
    print("2. 摄像头实时检测（推荐）")
    print("3. 视频文件检测")

    choice = input("请选择模式 (1-3): ").strip()

    if choice == '1':
        path = input("请输入图像路径（回车使用默认）: ").strip()
        predict_image(path if path else IMAGE_PATH)

    elif choice == '2':
        predict_camera()

    elif choice == '3':
        video_path = input("请输入视频路径: ").strip()
        predict_video(video_path)

    else:
        print("❌ 无效选择")
