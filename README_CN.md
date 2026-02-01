# AlphaGPT-Routine

AlphaGPT-Routine 是一个针对 A 股市场的自动化量化策略发现与执行工具。它利用基于 Transformer 的模型 (AlphaGPT) 通过波兰表示法 (Polish Notation) 发现并优化数学阿尔法公式，利用历史数据进行验证，并通过钉钉提供每日交易信号。

## 🚀 核心功能

- **阿尔法发现**：使用强大的 Transformer 模型自动化发现交易信号。
- **动态编码**：将人类可读的公式（例如 `ADD(RET, RET5)`）直接转换为模型 Token，反之亦然。
- **稳健回测**：严格的样本外验证，包含真实的交易成本和滑点模拟。
- **AkShare 集成**：无缝获取 A 股股票、指数和 ETF 数据。
- **钉钉通知**：每日自动报告持仓信号和业绩摘要。
- **云原生**：完全通过环境变量配置，支持 GitHub Actions 自动化。

## 🛠️ 配置参数（环境变量）

程序非常灵活，可以完全通过环境变量进行控制。

| 变量 | 描述 | 默认值 |
| :--- | :--- | :--- |
| `INDEX_CODE` | 股票或指数代码（例如 `600001`, `000300`） | `600001` |
| `CODE_FORMULA` | `代码:公式` 格式的组合覆盖 | - |
| `BEST_FORMULA` | 指定要使用的公式（跳过训练） | - |
| `START_DATE` | 数据开始日期 (YYYYMMDD) | `20240101` |
| `END_DATE` | 数据结束日期 (YYYYMMDD) | `20270101` |
| `TRAIN_ITERATIONS` | 训练轮数 | `100` |
| `BATCH_SIZE` | 模型批大小 | `1024` |
| `MAX_SEQ_LEN` | 公式最大复杂度 | `10` |
| `COST_RATE` | 交易费率（税费 + 手续费） | `0.0004` |
| `ONLY_LONG` | 是否仅支持多头市场（True/False） | `True` |
| `DINGTALK_WEBHOOK` | 钉钉机器人 Webhook 地址 | - |
| `DINGTALK_SECRET` | 钉钉机器人签名密钥 | - |

## 📦 GitHub Actions 设置

你可以利用 GitHub Actions 的每日例行任务免费运行此项目。

### 1. 配置 Secrets 和变量
进入你的 GitHub 仓库 -> **Settings** -> **Secrets and variables** -> **Actions**。

- **Secrets**（用于敏感数据）：
  - `DINGTALK_WEBHOOK`：你的机器人 Webhook 地址。
  - `DINGTALK_SECRET`：你的机器人安全密钥（签名）。
  - `CODE_FORMULA`：（可选）如果你想锁定特定的代码和公式。

- **Variables**（用于常规配置）：
  - `INDEX_CODE`：目标股票/指数。
  - `TRAIN_ITERATIONS`：策略发现设置为 `100`，每日报告可以设置为 `1`。
  - `ONLY_LONG`：A 股市场请设置为 `True`。

### 2. 手动触发
你可以在 **Actions** 选项卡中选择 "A-Stock AlphaGPT routine"，然后点击 **Run workflow** 手动触发工作流。你可以通过输入框覆盖 `CODE_FORMULA` 和日期。

## 💻 本地设置

```bash
# 安装依赖
pip install torch pandas numpy akshare matplotlib pyarrow tqdm requests

# 运行策略发现（默认设置）
python times_astock.py

# 使用自定义公式和钉钉运行
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/..."
export DINGTALK_SECRET="SEC..."
export CODE_FORMULA="600001:NEG(RET)"
python times_astock.py
```

## 📝 免责声明
本软件仅用于教育目的。生成的量化策略基于历史数据，不保证未来回报。股市交易涉及重大风险。
