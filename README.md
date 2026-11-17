# -YOLOV11-

基于 YOLOv11 目标检测模型构建课堂行为检测系统。系统由多源输入检测模块、行为分析模块、图像参数调节模块及功能控制模块组成。在模型层面，利用 YOLOv11 的高效特征提取与多方面检测能力完成课堂行为建模，实现对异常与正常行为的实时识别；在功能实现层面，系统支持摄像头、屏幕、图片与视频等多种检测方式，并结合置信度分析与行为状态判定完成结果可视化展示。

## 目录结构

```
.
├── yolo_to_coco.py                  # YOLO 格式标注 → COCO 格式转换脚本
├── data/
│   ├── classroom_behavior/          # 课堂行为数据集（6 类）
│   │   ├── images/                  # 图像（train / valid / test）
│   │   ├── labels/                  # YOLO 标签
│   │   ├── data.yaml                # 数据集配置
│   │   └── classroom_behavior.rar   # 数据集压缩包
│   └── class/                       # 课堂行为数据集（7 类）
│       ├── images/
│       ├── labels/
│       ├── VOC/                     # VOC 格式标注
│       ├── {train,valid,test}_coco.json
│       ├── data.yaml
│       └── class.rar
└── .gitattributes                   # 大文件通过 Git LFS 管理
```

## 数据集类别

- `classroom_behavior`（6 类）：Bend Over The Desk、Hand Under Table、Look Around、Normal、Stand Up、Wave
- `class`（7 类）：sleep、raise_head、using_computer、stand、using_phone、raise_hand、read

> 数据集为二进制大文件，通过 Git LFS 存储。克隆后请先安装 [Git LFS](https://git-lfs.com/)，再执行 `git lfs pull` 获取实际文件。

## 环境依赖

```bash
pip install opencv-python pyyaml tqdm ultralytics
```

## 使用方式

### YOLO 标注转 COCO 格式

`yolo_to_coco.py` 会读取 YOLO 数据集的 `data.yaml`，按 train / val / test 三个划分生成 COCO 格式的 `annotations/{split}.json`，并把对应图像复制到输出目录。

按当前脚本的默认配置运行：

```bash
python yolo_to_coco.py
```

默认参数（在脚本 `__main__` 中修改）：

```python
yolo_root   = r"data/class"
dataset_yaml = r"data/class/data.yaml"
output_dir   = r"data/class_coco"
```

也可在 Python 中直接调用函数：

```python
from yolo_to_coco import convert_yolo_to_coco

convert_yolo_to_coco(
    yolo_root="data/classroom_behavior",
    dataset_yaml="data/classroom_behavior/data.yaml",
    output_dir="data/classroom_behavior_coco",
)
```

### 路径说明

`data.yaml` 中的 `train` / `val` / `test` 路径为原始作者本机路径，迁移到其他机器时请改为相对本仓库的路径，例如：

```yaml
train: images/train
val: images/valid
test: images/test
```
