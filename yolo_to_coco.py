import json
import os
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def convert_yolo_to_coco(yolo_root, output_dir, dataset_yaml):
    """
    将YOLO格式数据集转换为COCO格式

    Args:
        yolo_root: YOLO数据集根目录
        output_dir: COCO输出目录
        dataset_yaml: YOLO数据集配置文件路径
    """

    # 读取YOLO数据集配置
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        yolo_config = yaml.safe_load(f)

    # 清空输出目录（如果选择）
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"已清空目录: {output_dir}")
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    # 获取类别信息
    categories = []
    for idx, name in yolo_config['names'].items():
        categories.append({
            'id': int(idx) + 1,  # COCO从1开始
            'name': name,
            'supercategory': 'none'
        })

    # 处理每个分割（train, val, test）
    splits = ['train', 'val', 'test']

    for split in splits:
        print(f"\n处理 {split} 数据集...")

        # 创建图像目录
        split_img_dir = os.path.join(output_dir, split)
        os.makedirs(split_img_dir, exist_ok=True)

        # 获取YOLO格式的路径
        yolo_img_dir = os.path.join(yolo_root, yolo_config[split])
        yolo_label_dir = os.path.join(yolo_root, 'labels', split)

        # 获取所有图像文件
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
        img_files = []
        for ext in img_extensions:
            img_files.extend(list(Path(yolo_img_dir).glob(f'*{ext}')))
            img_files.extend(list(Path(yolo_img_dir).glob(f'*{ext.upper()}')))

        coco_data = {
            'images': [],
            'annotations': [],
            'categories': categories
        }

        annotation_id = 1
        image_id = 1

        for img_path in tqdm(img_files, desc=f"处理 {split} 图像"):
            # 读取图像尺寸
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            height, width = img.shape[:2]

            # 创建COCO图像信息
            img_filename = img_path.name
            dst_img_path = os.path.join(split_img_dir, img_filename)

            # 复制图像
            shutil.copy2(img_path, dst_img_path)

            coco_data['images'].append({
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': img_filename,
                'license': 1,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': ''
            })

            # 读取对应的标签文件
            label_path = Path(yolo_label_dir) / f"{img_path.stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                    # 将YOLO格式转换为COCO格式（绝对坐标）
                    x = (x_center - bbox_width / 2) * width
                    y = (y_center - bbox_height / 2) * height
                    w = bbox_width * width
                    h = bbox_height * height

                    # 确保坐标在图像范围内
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    w = max(1, min(w, width - x))
                    h = max(1, min(h, height - y))

                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': int(class_id) + 1,  # COCO从1开始
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'area': float(w * h),
                        'segmentation': [],
                        'iscrowd': 0
                    })

                    annotation_id += 1

            image_id += 1

        # 保存COCO格式的标注文件
        coco_file = os.path.join(output_dir, 'annotations', f'{split}.json')
        with open(coco_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

        print(f"{split} 数据集: {len(coco_data['images'])} 张图像, {len(coco_data['annotations'])} 个标注")

    print(f"\n✅ 转换完成！COCO数据集保存在: {output_dir}")


if __name__ == "__main__":
    # 配置你的路径
    yolo_root = r"data/class"
    dataset_yaml = r"data/class/data.yaml"
    output_dir = r"data/class_coco"

    convert_yolo_to_coco(yolo_root, output_dir, dataset_yaml)
