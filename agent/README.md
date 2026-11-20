# 课堂行为检测 + LLM 分析 Agent

在原有 YOLO 检测项目基础上，增加一层 AI 分析：检测结果 → 大模型 → 中文行为分析报告。

## 文件结构

```
agent/
├── config.json        # 配置：选后端、填 key、填权重路径、调阈值
├── detector.py        # YOLO 检测层：图片 → 行为列表
├── llm_client.py      # AI 层：行为列表 → 中文分析报告
├── main.py            # 命令行入口
└── requirements.txt   # 依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

（ultralytics 会自动带上 torch / opencv，首次安装较大）

## 三种运行方式

**1. 跑通链路（不联网、不需要权重）——先跑这个验证**
```bash
python main.py --mock
```
用内置假数据，会把实际发给 LLM 的 prompt 完整打印出来，并返回模拟分析报告。

**2. 只检测，不调用 AI**
```bash
python main.py --image path/to/photo.jpg --no-llm
```
默认已指向新训练的 7 类模型 `../YOLOvx/runs/detect/class_50ep/weights/best.pt`，无需额外放置权重。

**3. 完整链路：检测 + AI 分析（当前 LLM 为 mock，返回模拟报告）**
```bash
python main.py --image path/to/photo.jpg --save
```
`--save` 会把报告存到 `reports/` 目录。

> 用 anaconda 环境运行：`C:\ProgramData\anaconda3\python.exe main.py ...`（该环境已装 ultralytics；mock 模式不需要 openai 库）。代码兼容 Python 3.8。

## 切换 LLM 后端

改 `config.json` 的 `llm.provider`：

- `mock`：不联网，仅测试
- `openai_compatible`：任何 OpenAI 格式的接口，填好 `base_url` / `model` 即可。
  - 官方 OpenAI：`https://api.openai.com/v1`
  - 中转站 / Ollama / vLLM：换成对应地址（Ollama 一般是 `http://localhost:11434/v1`）

> 注意：`D:\DEEPSEEK` 那套是 QQ 机器人的私有网关，不是 OpenAI 兼容接口，不能直接填 here。

## API 密钥安全（重要）

**密钥绝不写进 `config.json`**——该文件会随仓库公开。仓库里的 `api_key` 恒为空。
运行时通过环境变量提供，代码优先读取环境变量：

```bash
# Windows CMD
set OPENAI_API_KEY=你的key
# Windows PowerShell
$env:OPENAI_API_KEY="你的key"
```

也可以设置 `OPENAI_BASE_URL` 覆盖接口地址。若两者都缺且 provider 为 `openai_compatible`，
程序会直接报错并提示，不会用空 key 悄悄联网。想在本地长期保存又不入库，可新建
`config.local.json`（已加入 `.gitignore`）。

## 权重与类别

`config.json` 的 `detector.weights` 默认指向已训练好的 7 类模型：`../YOLOvx/runs/detect/class_50ep/weights/best.pt`（相对 agent 目录）。
类别名自动从权重读取，共 7 类：`sleep / raise_head / using_computer / stand / using_phone / raise_hand / read`。
换模型只需改 `detector.weights` 路径，不用改代码。

## 接入真实 LLM（可选）

目前 `llm.provider = mock`，返回模拟报告且会把真实 prompt 打印出来供检查。
拿到任何 OpenAI 兼容接口后，把 `config.json` 的 provider 改为 `openai_compatible`、
填好 `base_url` / `model`，并用**环境变量**提供密钥（见上一节），安装 `pip install openai`
（mock 模式不需要），即可输出真实 AI 分析报告。
