# Candlestick ViT — Stock Trend Prediction using Vision Transformer & XAI

> **Group 1 Assignment** | Deep Learning Framework for Stock Trend Prediction via Candlestick Chart Image Classification

---

## 📌 Project Overview

This project transforms financial time-series data into candlestick chart images and frames stock trend prediction as an **image classification problem**. A **Vision Transformer (ViT)** is trained on these images, and **Explainable AI (XAI)** techniques are applied to interpret what patterns the model learns.

**Target Market**: NSE (National Stock Exchange of India)  
**Data Source**: Yahoo Finance API via `yfinance`  
**Interval**: 1-hour OHLC candlesticks  

---

## ✅ Progress Tracker

| Phase | Task | Status |
|-------|------|--------|
| 1 | Data Collection & Preprocessing | ✅ Done |
| 2 | Labeling Strategy | ✅ Done |
| 3 | Dataset Preparation | ✅ Done |
| 4 | Model Implementation (ViT) | 🔲 Pending |
| 5 | Explainability (XAI / Attention / Grad-CAM) | 🔲 Pending |
| 6 | Evaluation & Analysis | 🔲 Pending |
| 7 | Comparative Study | 🔲 Pending |
| 8 | Final Report | 🔲 Pending |

---

## 📁 Repository Structure

```
candlestick-vit/
│
├── Data_Collection_and_Preprocessing_take_3.ipynb   # Phase 1: Fetch & preprocess OHLC data
├── Labelling_Strategy.ipynb                          # Phase 2: Label generation pipeline
├── Dataset_Preparation.ipynb                         # Phase 3: Train/val/test split & folder org
│
├── stock_dataset/
│   ├── raw_images/                  # Candlestick images organized by stock
│   │   ├── AARTIIND/
│   │   ├── AFFLE/
│   │   ├── ALKEM/
│   │   └── ... (50 stocks total)
│   └── labels.csv                   # Master label file (18,867 entries)
│
└── README.md
```

---

## 📊 Stock Universe

The dataset covers **50 stocks** across NSE, spanning three market cap categories.

### 🔵 Large Cap

| Symbol | Company | Sector |
|--------|---------|--------|
| RELIANCE | Reliance Industries | Energy |
| HDFCBANK | HDFC Bank | Financial Services |
| TCS | Tata Consultancy Services | IT |
| ICICIBANK | ICICI Bank | Financial Services |
| INFY | Infosys | IT |
| HINDUNILVR | Hindustan Unilever | FMCG |
| ITC | ITC Limited | Diversified |
| SBIN | State Bank of India | Financial Services |
| BHARTIARTL | Bharti Airtel | Telecom |
| KOTAKBANK | Kotak Mahindra Bank | Financial Services |
| AXISBANK | Axis Bank | Financial Services |
| LT | Larsen & Toubro | Engineering |
| ULTRACEMCO | UltraTech Cement | Cement |
| MARUTI | Maruti Suzuki | Automotive |

### 🟡 Mid Cap

| Symbol | Company | Sector |
|--------|---------|--------|
| LTTS | LT Technology Services | IT Services |
| VOLTAS | Voltas | HVAC & Cooling |
| TATAELXSI | Tata Elxsi | IT Services |
| ASHOKLEY | Ashok Leyland | Automotive |
| MUTHOOTFIN | Muthoot Finance | Financial Services |
| CUMMINSIND | Cummins India | Industrial |
| DATAPATTNS | Data Patterns | Defense |
| FEDERALBNK | Federal Bank | Financial Services |
| MPHASIS | Mphasis | IT Services |
| BIOCON | Biocon | Pharma |
| LTIM | LTIMindtree | IT Services |
| SUNPHARMA | Sun Pharmaceutical | Pharma |
| PERSISTENT | Persistent Systems | IT Services |
| COFORGE | Coforge | IT Services |
| PAGEIND | Page Industries | FMCG |
| POLYCAB | Polycab India | Electricals |
| LAURUSLABS | Laurus Labs | Pharma |
| DEEPAKNTR | Deepak Nitrite | Chemicals |
| BHARATFORG | Bharat Forge | Automotive |
| ALKEM | Alkem Laboratories | Pharma |
| APLAPOLLO | APL Apollo Tubes | Steel |
| ASTRAL | Astral Ltd | Plumbing |
| PIIND | PI Industries | Agro-Chemicals |
| SRF | SRF Limited | Chemicals |
| TORNTPHARM | Torrent Pharmaceuticals | Pharma |

### 🔴 Small Cap

| Symbol | Company | Sector |
|--------|---------|--------|
| IRCTC | Indian Railway Catering & Tourism | Travel & Hospitality |
| CDSL | Central Depository Services | Financial Services |
| NAZARA | Nazara Technologies | Gaming & Media |
| ROUTE | Route Mobile | Telecom |
| TANLA | Tanla Platforms | Telecom |
| AFFLE | Affle (India) | Digital Marketing |
| AARTIIND | Aarti Industries | Chemicals |
| IRCON | IRCON International | Construction |
| RVNL | Rail Vikas Nigam | Infrastructure |
| AUBANK | AU Small Finance Bank | Financial Services |
| KNRCON | KNR Constructions | Infrastructure |

---

## ✅ Phase 1: Data Collection & Preprocessing

**Notebook**: `Data_Collection_and_Preprocessing_take_3.ipynb`

### What was done
- Fetched **60 days** of **1-hour interval** OHLC data for all 50 stocks using `yfinance`
- Converted timestamps from UTC → `Asia/Kolkata` timezone
- Filtered to **NSE market hours only**: `09:15 AM – 3:30 PM IST`
- Retained only `Open`, `High`, `Low`, `Close` columns
- Generated **candlestick chart images** (20-candle sliding window, step size = 1) per stock

### Parameters

```python
PERIOD    = "60d"
INTERVAL  = "1h"
TIMEZONE  = "Asia/Kolkata"
MARKET_START = time(9, 15)
MARKET_END   = time(15, 30)
WINDOW_SIZE  = 20   # ~2.5 trading days per image
```

### Output
- Raw candlestick images saved under `stock_dataset/raw_images/<STOCK_NAME>/`
- Naming convention: `<STOCK_NAME>_<window_index>.png`

---

## ✅ Phase 2: Labeling Strategy

**Notebook**: `Labelling_Strategy.ipynb`

### Methodology

Each candlestick image window is labeled based on the **next trading day's close-to-close return**:

| Label | Condition | Interpretation |
|-------|-----------|----------------|
| **Up** | Return > +0.5% | Bullish — price expected to rise |
| **Down** | Return < -0.5% | Bearish — price expected to fall |
| **Neutral** | −0.5% ≤ Return ≤ +0.5% | Consolidation — direction unclear |

**Return formula**:
```
return = (close_next_trading_day - close_today) / close_today
```

### Pipeline

```
1. Fetch 60d × 1h OHLC per stock (yfinance)
2. Timezone convert + market hours filter
3. Sliding window (size=20, step=1) over candles
4. For each window → find next trading day's close
5. Calculate return → apply threshold → assign label
6. Save: image_path | label | stock | timestamp → labels.csv
```

### Label Distribution (50 stocks, 60 days)

| Label | Count | Percentage |
|-------|-------|-----------|
| **Down** | 8,729 | 46.27% |
| **Up** | 6,458 | 34.23% |
| **Neutral** | 3,680 | 19.50% |
| **Total** | **18,867** | 100% |

> **Note**: Dataset exhibits a natural bearish bias. Class weights or stratified sampling are recommended during training.

### Suggested Class Weights for Training

```python
class_weights = {
    "down":    0.67,
    "neutral": 1.28,
    "up":      0.93
}
```

### Threshold Sensitivity

| Threshold | Up % | Neutral % | Down % | Use Case |
|-----------|------|-----------|--------|----------|
| ±0.3% | 29% | 8% | 63% | Conservative |
| **±0.5%** | **34%** | **20%** | **46%** | **Default (used)** |
| ±1.0% | 38% | 35% | 27% | Aggressive |

### Output
- `stock_dataset/labels.csv` — columns: `image_path`, `label`, `stock`, `timestamp`
- ~20 seconds execution time for all 50 stocks

---

## ✅ Phase 3: Dataset Preparation

**Notebook**: `Dataset_Preparation.ipynb`

### Splits

| Split | Percentage | Approx. Samples |
|-------|-----------|-----------------|
| Training | 70% | ~13,207 |
| Validation | 20% | ~3,773 |
| Testing | 10% | ~1,887 |

> **Note**: Assignment specifies 70/20/10; labeling strategy docs use 70/15/15 — the notebook implements 70/20/10 per assignment spec.

### Key Considerations
- **Split by date or stock** (not random) to prevent data leakage
- Stratified splitting to maintain class distribution across splits
- Final folder structure:

```
stock_dataset/
├── train/
│   ├── up/
│   ├── down/
│   └── neutral/
├── val/
│   ├── up/
│   ├── down/
│   └── neutral/
└── test/
    ├── up/
    ├── down/
    └── neutral/
```

---

## 🔲 Phase 4: Model Implementation (ViT) — Pending

### Plan
- Implement **Vision Transformer (ViT)** using PyTorch
- Use pretrained weights (e.g., `google/vit-base-patch16-224` from HuggingFace)
- Fine-tune on candlestick image dataset
- Handle class imbalance via weighted loss or oversampling
- Hyperparameter tuning: learning rate, batch size, patch size, number of heads

### Baseline Target
- Majority class baseline: **46.3%** (always predict "Down")
- Practical target: **> 55% accuracy**

---

## 🔲 Phase 5: Explainability (XAI) — Pending

### Plan
- **Attention Map Visualization** — visualize which patches the ViT attends to
- **Grad-CAM** — highlight important image regions per prediction class
- Analyze whether model focuses on known candlestick patterns (wicks, bodies, gaps)

---

## 🔲 Phase 6: Evaluation & Analysis — Pending

### Metrics
- Accuracy, Precision, Recall, F1-score (per class + macro/weighted)
- Confusion matrix
- Class-wise performance breakdown

---

## 🔲 Phase 7: Comparative Study — Pending

- Compare ViT predictions against classical candlestick patterns (Doji, Hammer, Engulfing)
- Discuss whether the model rediscovers known patterns or identifies new structures

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Data Fetching | `yfinance` |
| Preprocessing | `pandas`, `numpy`, `pytz` |
| Image Generation | `matplotlib`, `mplfinance` |
| Model | `PyTorch`, `transformers` (HuggingFace ViT) |
| XAI | `pytorch-grad-cam`, attention rollout |
| Environment | Google Colab (T4 GPU) |

---

## 🔄 Reproducibility

All notebooks are designed to run end-to-end on **Google Colab** with Google Drive mounted at `/content/drive/MyDrive/stock_dataset/`.

```bash
# Execution order
1. Data_Collection_and_Preprocessing_take_3.ipynb   # Fetch data + generate images
2. Labelling_Strategy.ipynb                          # Generate labels.csv
3. Dataset_Preparation.ipynb                         # Create train/val/test splits
```

---

## 📝 Notes

- All stock symbols use NSE notation (suffix `.NS`) for yfinance queries
- Trading hours: 09:15 AM – 3:30 PM IST (Monday–Friday)
- Holiday calendar: NSE holidays automatically handled by yfinance
- Lookahead bias is avoided — labels use only the **next** trading day's close

---

**Last Updated**: April 2026  
**Total Stocks**: 50  
**Total Labeled Samples**: 18,867  
**Data Format**: Candlestick images (PNG) + OHLCV via yfinance  
**Label File**: `stock_dataset/labels.csv`
