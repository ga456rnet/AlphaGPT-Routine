# AlphaGPT-AStock

AlphaGPT-AStock is an automated quantitative strategy discovery and execution tool for the A-share market. It utilizes a Transformer-based model (AlphaGPT) to discover and optimize mathematical alpha formulas using Polish Notation, verifies them against historical data, and provides daily trading signals via DingTalk.

## 🚀 Key Features

- **Alpha Discovery**: Automated discovery of trading signals using a powerful Transformer model.
- **Dynamic Encoding**: Convert human-readable formulas (e.g., `ADD(RET, RET5)`) directly into model tokens and vice versa.
- **Robust Backtesting**: Strict out-of-sample verification with realistic transaction costs and slippage modeling.
- **AkShare Integration**: Seamless data fetching for A-share stocks, indices, and ETFs.
- **DingTalk Notifications**: Automated daily reports with position signals and performance summary.
- **Cloud Native**: Fully configurable via environment variables and ready for GitHub Actions automation.

## 🛠️ Configuration (Environment Variables)

The program is highly flexible and can be controlled entirely through environment variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `INDEX_CODE` | Stock or Index code (e.g., `600001`, `000300`) | `600001` |
| `CODE_FORMULA` | Combined override in `CODE:FORMULA` format | - |
| `BEST_FORMULA` | Specific formula to use (skips training) | - |
| `START_DATE` | Data start date (YYYYMMDD) | `20240101` |
| `END_DATE` | Data end date (YYYYMMDD) | `20270101` |
| `TRAIN_ITERATIONS` | Number of training epochs | `100` |
| `BATCH_SIZE` | Model batch size | `1024` |
| `MAX_SEQ_LEN` | Maximum complexity of the formula | `10` |
| `COST_RATE` | Transaction cost (taxes + fees) | `0.0004` |
| `ONLY_LONG` | Support for long-only market (True/False) | `True` |
| `DINGTALK_WEBHOOK` | DingTalk Bot Webhook URL | - |
| `DINGTALK_SECRET` | DingTalk Bot Signature Secret | - |

## 📦 GitHub Actions Setup

You can run this project for free using GitHub Actions daily routine.

### 1. Configure Secrets & Variables
Go to your GitHub repository -> **Settings** -> **Secrets and variables** -> **Actions**.

- **Secrets** (for sensitive data):
  - `DINGTALK_WEBHOOK`: Your bot's webhook URL.
  - `DINGTALK_SECRET`: Your bot's security secret (Signature).
  - `CODE_FORMULA`: (Optional) If you want to lock a specific code and formula.

- **Variables** (for general config):
  - `INDEX_CODE`: Target stock/index.
  - `TRAIN_ITERATIONS`: Set to `100` for discovery or `1` for daily reporting.
  - `ONLY_LONG`: Set to `True` for A-shares.

### 2. Manual Trigger
You can manually trigger the workflow from the **Actions** tab by selecting "A-Stock AlphaGPT routine" and clicking **Run workflow**. You can override the `CODE_FORMULA` and dates via the input fields.

## 💻 Local Setup

```bash
# Install dependencies
pip install torch pandas numpy akshare matplotlib pyarrow tqdm requests

# Run discovery (default settings)
python times_astock.py

# Run with custom formula and DingTalk
export DINGTALK_WEBHOOK="https://oapi.dingtalk.com/..."
export DINGTALK_SECRET="SEC..."
export CODE_FORMULA="600001:NEG(RET)"
python times_astock.py
```

## 📝 Disclaimer
This software is for educational purposes only. The quantitative strategies generated are based on historical data and do not guarantee future returns. Trading in the stock market involves significant risk.
