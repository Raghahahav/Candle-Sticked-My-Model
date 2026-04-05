# Stock Dataset

This project includes comprehensive stock data analysis and modeling across three market capitalizations of the Indian stock market (NSE).

## Stock Universe

Our analysis covers **38 stocks** across three distinct market cap categories:

---

##  Large Cap (11 stocks)

Large-cap stocks represent the most established and liquid companies in the Indian market. These are the blue-chip stocks with significant market presence and trading volumes.

| Symbol | Company | Sector |
|--------|---------|--------|
| RELIANCE.NS | Reliance Industries | Energy |
| HDFCBANK.NS | HDFC Bank | Financial Services |
| TCS.NS | Tata Consultancy Services | IT |
| ICICIBANK.NS | ICICI Bank | Financial Services |
| INFY.NS | Infosys | IT |
| HINDUNILVR.NS | Hindustan Unilever | FMCG |
| ITC.NS | ITC Limited | Diversified |
| SBIN.NS | State Bank of India | Financial Services |
| BHARTIARTL.NS | Bharti Airtel | Telecom |
| KOTAKBANK.NS | Kotak Mahindra Bank | Financial Services |
| AXISBANK.NS | Axis Bank | Financial Services |

**Characteristics:**
- Market Cap: ₹1,00,000 Cr+
- High liquidity and trading volumes
- Stable, established companies
- Lower volatility
- Suitable for long-term, conservative portfolios

---

##  Mid Cap (13 stocks)

Mid-cap stocks offer a balance between growth potential and stability. These companies have established market presence but offer more growth opportunities compared to large-cap stocks.

| Symbol | Company | Sector |
|--------|---------|--------|
| LTTS.NS | LT Infotech | IT Services |
| VOLTAS.NS | Voltas | HVAC & Cooling |
| TATAELXSI.NS | Tata Elxsi | IT Services |
| ASHOKLEY.NS | Ashok Leyland | Automotive |
| MUTHOOTFIN.NS | Muthoot Finance | Financial Services |
| CUMMINSIND.NS | Cummins India | Automotive |
| DATAPATNS.NS | Data Patterns | Defense |
| FEDERALBNK.NS | Federal Bank | Financial Services |
| MPHASIS.NS | Mphasis | IT Services |
| BIOCON.NS | Biocon | Pharma |
| LTIM.NS | LTIMindtree | IT Services |
| MARUTI.NS | Maruti Suzuki | Automotive |
| SUNPHARMA.NS | Sun Pharmaceutical | Pharma |

**Characteristics:**
- Market Cap: ₹20,000 - ₹1,00,000 Cr
- Moderate liquidity
- Growth-oriented companies
- Moderate to high volatility
- Suitable for growth-focused portfolios

---

##   Small Cap (14 stocks)

Small-cap stocks represent emerging companies with high growth potential but also higher risk. These stocks typically have lower market caps and trading volumes, offering significant upside opportunities.

| Symbol | Company | Sector |
|--------|---------|--------|
| IRCTC.NS | Indian Railway Catering & Tourism | Travel & Hospitality |
| CDSL.NS | Central Depository Services | Financial Services |
| NAZARA.NS | Nazara Technologies | Gaming & Media |
| ROUTE.NS | Route Mobile | Telecom |
| TANLA.NS | Tanla Platforms | Telecom |
| AFFLE.NS | Affle (India) | Digital Marketing |
| AARTIIND.NS | Aarti Industries | Chemicals |
| IRCON.NS | IRCON International | Construction |
| RVNL.NS | Rail Vikas Nigam | Infrastructure |
| AUBANK.NS | AU Small Finance Bank | Financial Services |
| ULTRACEMCO.NS | UltraTech Cement | Cement |
| KNRCON.NS | Konark Institute | Infrastructure |
| LT.NS | Larsen & Toubro | Engineering |
| KNRCON.NS | Konark Infrastructure | Infrastructure |

**Characteristics:**
- Market Cap: < ₹20,000 Cr
- Lower trading volumes
- High growth potential
- High volatility
- Higher risk-reward profile
- Suitable for aggressive, growth-focused portfolios

---

## 📊 Sector Distribution

```
IT Services & Software       : 5 stocks
Financial Services           : 7 stocks
Pharmaceuticals             : 3 stocks
Chemicals                   : 1 stock
Automotive                  : 3 stocks
FMCG & Consumer            : 1 stock
Energy                      : 1 stock
Telecom                     : 3 stocks
Engineering & Infrastructure: 3 stocks
Gaming & Media              : 1 stock
Defense                     : 1 stock
Travel & Hospitality        : 1 stock
Digital Marketing           : 1 stock
Construction                : 1 stock
Cement                      : 1 stock
```

---

## 🎯 Usage Guidelines

### For Machine Learning Models:
- **Classification**: Use market cap category as target variable for cap prediction models
- **Clustering**: Analyze stock behavior patterns within each cap category
- **Time Series**: Leverage different volatility profiles across categories

### For Portfolio Construction:
- **Conservative**: Allocate 70% Large Cap, 30% Mid Cap
- **Balanced**: Allocate 40% Large Cap, 40% Mid Cap, 20% Small Cap
- **Aggressive**: Allocate 30% Large Cap, 40% Mid Cap, 30% Small Cap

### For Risk Analysis:
- Large Cap: Lower drawdowns, suitable for risk management
- Mid Cap: Balanced risk-return profile
- Small Cap: High potential returns but requires careful risk management

---

## 📈 Key Statistics

| Metric | Large Cap | Mid Cap | Small Cap |
|--------|-----------|---------|-----------|
| Count | 11 | 13 | 14 |
| Avg Volatility | Low | Moderate | High |
| Liquidity | Very High | High | Moderate |
| Market Cap Range | ₹1L+ Cr | ₹20K-₹1L Cr | <₹20K Cr |
| Risk Profile | Low | Medium | High |
| Growth Potential | Stable | Good | Excellent |

---

## 🔄 Data Updates

All stock data is fetched from:
- **NSE (National Stock Exchange of India)**
- **Yahoo Finance API**
- **yfinance** Python library for automated updates

Recommended update frequency:
- Large Cap: Daily
- Mid Cap: Daily
- Small Cap: Daily to Weekly

---

## 📝 Notes

1. **Categorization** is based on market capitalization as of the dataset creation date
2. **Stock categories may change** if companies move between cap segments
3. **All symbols use NSE notation** (suffix `.NS`)
4. **Trading hours**: 09:15 AM - 3:30 PM IST (Monday - Friday)
5. **Holiday calendar**: Follows NSE holidays

---


**Last Updated**: 5 April 2026  
**Total Stocks**: 38  
**Data Format**: High-resolution candlestick images + OHLCV data  
