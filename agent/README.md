# 课堂行为检测 + DeepSeek AI 分析 Agent

YOLO 检测课堂行为 → 组装结构化 prompt → DeepSeek 大模型 → 中文行为分析报告。

## 文件结构

```
agent/
├── config.json        # 配置：DeepSeek 接口、模型、检测阈值
├── api_key.txt        # 【本地私有】DeepSeek 密钥，已 gitignore，绝不入库
├── 运行分析.bat       # 一键启动（把图片拖上去即可）
├── detector.py        # YOLO 检测层：图片 → 行为列表
├── llm_client.py      # AI 层：行为列表 → 中文分析报告
├── main.py            # 命令行入口
└── requirements.txt   # 依赖
```

## 快速开始（3 步）

**第 1 步：申请 DeepSeek 密钥**

打开 https://platform.deepseek.com/api_keys 注册并创建 API key（按量计费，极便宜，新用户有免费额度）。

**第 2 步：填入密钥**

用记事本打开 `agent/api_key.txt`，把里面的占位文字替换成你的 key（`sk-` 开头），保存。
这个文件已加入 `.gitignore`，不会被提交到 GitHub。

**第 3 步：运行**

- **最简单**：把一张图片直接拖到 `运行分析.bat` 上，自动检测 + AI 分析 + 保存报告。
- 或命令行：
  ```bash
  cd agent
  C:\ProgramData\anaconda3\python.exe main.py --image path/to/photo.jpg --save
  ```

报告会打印在终端，并用 `--save` 存到 `reports/` 目录。

## 其他运行方式

```bash
# 不联网，用假数据跑通链路（验证代码用，不需要 key）
python main.py --mock

# 只检测不调用 AI
python main.py --image path/to/photo.jpg --no-llm
```

> 注意：anaconda 的 Python 缺 SSL 路径，**必须**先执行
> `set PATH=C:\ProgramData\anaconda3\Library\bin;%PATH%`（bat 脚本已自动处理）。
> 手动在命令行跑时记得加这句，否则调 API 会报 SSL 错误。

## 配置说明

`config.json` 已预置 DeepSeek：

- `base_url`: `https://api.deepseek.com/v1`
- `model`: `deepseek-chat`（省钱，够用）；想要更强推理可换 `deepseek-reasoner`
- `provider`: `openai_compatible`；改回 `mock` 则不联网只测试

密钥读取优先级：**环境变量 `OPENAI_API_KEY` > `api_key.txt` > config.json 占位**。
换其他 OpenAI 兼容接口（通义、Kimi、中转站等）只需改 `base_url` 和 `model`。

## 权重与类别

`detector.weights` 默认指向 7 类模型 `../YOLOvx/runs/detect/class_50ep/weights/best.pt`（相对 agent 目录）。
类别：`sleep / raise_head / using_computer / stand / using_phone / raise_hand / read`。
换模型只改路径，不改代码。

## 安全提醒

`api_key.txt` 和 `config.local.json` 已在 `.gitignore` 中，**永远不要**把真实 key 写进
`config.json` 或任何会提交到仓库的文件。密钥只放本地文件或用环境变量提供。
