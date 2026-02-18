# E-Commerce Sales & Customer Analytics

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![SQL](https://img.shields.io/badge/SQL-MySQL-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## Project Overview

End-to-end analysis of **100,000+ e-commerce transactions** from a Brazilian
marketplace to uncover revenue trends, customer behavior patterns, and
delivery performance insights.

## Key Findings

| KPI                   | Value     |
| --------------------- | --------- |
| Total Revenue         | R$ 13.5M+ |
| Total Orders          | 96,000+   |
| Unique Customers      | 93,000+   |
| Average Order Value   | R$ 154    |
| On-Time Delivery Rate | ~93%      |
| Customer Satisfaction | 4.07/5.0  |
| Repeat Customer Rate  | ~3%       |

## Analysis Performed

1. **Revenue Analysis** - Monthly trends, day-of-week & hourly patterns
2. **Product Analysis** - Top categories, revenue drivers, satisfaction scores
3. **Customer Segmentation** - RFM analysis (Recency, Frequency, Monetary)
4. **Delivery Performance** - On-time rates, state-wise delivery times
5. **Payment Analysis** - Payment method preferences & patterns
6. **Seller Performance** - Top sellers, revenue concentration
7. **Correlation Analysis** - Relationships between key variables

## Tools & Technologies

- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Plotly)
- **SQL** (Complex queries, CTEs, Window Functions)
- **Excel** (Pivot Tables, Summary Reports)
- **Tableau** (Interactive Dashboard)

## Project Structure

```
├── data/                          # Raw CSV files (from Kaggle)
├── ecommerce_analysis.py          # Main analysis script
├── sql_queries.sql                # All SQL queries
├── ecommerce_analysis_summary.xlsx # Excel summary
├── rfm_customer_segments.csv      # RFM output
├── *.png                          # All visualizations
├── business_report.md             # Business recommendations
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Sample Visualizations

### Monthly Revenue Trend

![Revenue](01_monthly_revenue_trend.png)

### RFM Customer Segmentation

![RFM](08_rfm_segmentation.png)

### Delivery Performance

![Delivery](09_delivery_analysis.png)

## Business Recommendations

1. **Improve Repeat Rate (currently ~3%)** → Implement loyalty program
   targeting "Potential Loyalists" segment
2. **Optimize Delivery in Northern States** → Average delivery 25+ days
   vs 10 days in São Paulo
3. **Focus on Top Categories** → Top 5 categories drive 45%+ revenue
4. **Reduce Late Deliveries** → Late orders score 2.5⭐ vs 4.3⭐
   for on-time (direct impact on satisfaction)
5. **Peak Hour Marketing** → 70% of orders placed between 10AM-10PM,
   optimize ad spend for these hours

## Dataset

- **Source:** [Brazilian E-Commerce (Olist) - Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Records:** 100K+ orders (2016-2018)
- **Tables:** 8 interconnected datasets

## How to Run

```bash
git clone https://github.com/tanmay121-hub/ecommerce-sales-analytics.git

cd ecommerce-sales-analytics

pip install -r requirements.txt

python ecommerce_analysis.py
```

## Contact

- **LinkedIn:** [Tanmay Patil](https://www.linkedin.com/in/tanmay-patil-10997a259/)
- **Email:** tanmaypatil.dev@gmail.com
