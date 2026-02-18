
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 60)
print("  E-COMMERCE SALES & CUSTOMER ANALYTICS")
print("=" * 60)

print("\n Loading datasets...")

orders = pd.read_csv('data/olist_orders_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
customers = pd.read_csv('data/olist_customers_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')
sellers = pd.read_csv('data/olist_sellers_dataset.csv')
category_translation = pd.read_csv('data/product_category_name_translation.csv')

print(" All 8 datasets loaded successfully!")
print(f"   Orders: {orders.shape[0]:,} rows")
print(f"   Order Items: {order_items.shape[0]:,} rows")
print(f"   Products: {products.shape[0]:,} rows")
print(f"   Customers: {customers.shape[0]:,} rows")
print(f"   Payments: {payments.shape[0]:,} rows")
print(f"   Reviews: {reviews.shape[0]:,} rows")


print("\n Cleaning data...")

# Convert date columns
date_columns = [
    'order_purchase_timestamp',
    'order_approved_at',
    'order_delivered_carrier_date',
    'order_delivered_customer_date',
    'order_estimated_delivery_date'
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col])

# Merge datasets into one master dataframe
df = orders.merge(order_items, on='order_id', how='left')
df = df.merge(products, on='product_id', how='left')
df = df.merge(customers, on='customer_id', how='left')
df = df.merge(payments, on='order_id', how='left')
df = df.merge(category_translation, on='product_category_name', how='left')
df = df.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')

# Remove duplicates
df = df.drop_duplicates()

# Fill missing category names
df['product_category_name_english'] = df['product_category_name_english'].fillna('unknown')

# Add useful columns
df['order_year'] = df['order_purchase_timestamp'].dt.year
df['order_month'] = df['order_purchase_timestamp'].dt.month
df['order_year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
df['order_day_of_week'] = df['order_purchase_timestamp'].dt.day_name()
df['order_hour'] = df['order_purchase_timestamp'].dt.hour

# Calculate delivery time
df['delivery_days'] = (
    df['order_delivered_customer_date'] - df['order_purchase_timestamp']
).dt.days

# Calculate if delivery was late
df['estimated_delivery_days'] = (
    df['order_estimated_delivery_date'] - df['order_purchase_timestamp']
).dt.days

df['delivery_status'] = np.where(
    df['order_delivered_customer_date'] <= df['order_estimated_delivery_date'],
    'On Time', 'Late'
)

# Check missing values
print(f"   Total records after merge: {df.shape[0]:,}")
print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Filter only delivered orders for analysis
df_delivered = df[df['order_status'] == 'delivered'].copy()
print(f"   Delivered orders for analysis: {df_delivered.shape[0]:,}")
print(" Data cleaning complete!")

# OVERVIEW KPIs
print("\n KEY PERFORMANCE INDICATORS (KPIs)")
print("-" * 50)

total_revenue = df_delivered['payment_value'].sum()
total_orders = df_delivered['order_id'].nunique()
total_customers = df_delivered['customer_unique_id'].nunique()
avg_order_value = total_revenue / total_orders
avg_review_score = df_delivered['review_score'].mean()
total_products_sold = df_delivered['product_id'].nunique()

print(f"    Total Revenue:        R$ {total_revenue:,.2f}")
print(f"    Total Orders:         {total_orders:,}")
print(f"    Total Customers:      {total_customers:,}")
print(f"    Avg Order Value:      R$ {avg_order_value:,.2f}")
print(f"    Avg Review Score:     {avg_review_score:.2f} / 5.0")
print(f"    Unique Products Sold: {total_products_sold:,}")

# REVENUE ANALYSIS
print("\n Generating Revenue Analysis...")

# Monthly Revenue Trend
monthly_revenue = (
    df_delivered
    .groupby('order_year_month')
    .agg(
        revenue=('payment_value', 'sum'),
        orders=('order_id', 'nunique'),
        customers=('customer_unique_id', 'nunique')
    )
    .reset_index()
)
monthly_revenue['order_year_month'] = monthly_revenue['order_year_month'].astype(str)

fig, ax1 = plt.subplots(figsize=(14, 6))
color1 = '#2196F3'
color2 = '#FF9800'

ax1.bar(monthly_revenue['order_year_month'], monthly_revenue['revenue'],
        color=color1, alpha=0.7, label='Revenue')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue (R$)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
plt.xticks(rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(monthly_revenue['order_year_month'], monthly_revenue['orders'],
         color=color2, marker='o', linewidth=2, label='Orders')
ax2.set_ylabel('Number of Orders', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(' Monthly Revenue & Order Trends', fontsize=16, fontweight='bold')
fig.tight_layout()
plt.savefig('01_monthly_revenue_trend.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 01_monthly_revenue_trend.png")

# Revenue by Day of Week
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_revenue = (
    df_delivered
    .groupby('order_day_of_week')['payment_value']
    .agg(['sum', 'count'])
    .reindex(dow_order)
    .reset_index()
)
daily_revenue.columns = ['day', 'revenue', 'orders']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(daily_revenue['day'], daily_revenue['revenue'], color='#4CAF50', alpha=0.8)
axes[0].set_title('Revenue by Day of Week', fontweight='bold')
axes[0].set_ylabel('Revenue (R$)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(daily_revenue['day'], daily_revenue['orders'], color='#9C27B0', alpha=0.8)
axes[1].set_title('Orders by Day of Week', fontweight='bold')
axes[1].set_ylabel('Number of Orders')
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle(' Sales by Day of Week', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_day_of_week_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 02_day_of_week_analysis.png")

# Revenue by Hour 
hourly = df_delivered.groupby('order_hour')['payment_value'].agg(['sum', 'count']).reset_index()
hourly.columns = ['hour', 'revenue', 'orders']

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(hourly['hour'], hourly['orders'], alpha=0.3, color='#E91E63')
ax.plot(hourly['hour'], hourly['orders'], color='#E91E63', linewidth=2, marker='o')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Number of Orders')
ax.set_title(' Orders by Hour of Day', fontsize=16, fontweight='bold')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig('03_hourly_orders.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 03_hourly_orders.png")

# PRODUCT ANALYSIS
print("\n Generating Product Analysis...")

# Top 15 Product Categories by Revenue
category_revenue = (
    df_delivered
    .groupby('product_category_name_english')
    .agg(
        revenue=('payment_value', 'sum'),
        orders=('order_id', 'nunique'),
        avg_review=('review_score', 'mean')
    )
    .sort_values('revenue', ascending=False)
    .head(15)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(
    category_revenue['product_category_name_english'],
    category_revenue['revenue'],
    color=plt.cm.viridis(np.linspace(0.3, 0.9, 15))
)
ax.invert_yaxis()
ax.set_xlabel('Revenue (RS)')
ax.set_title(' Top 15 Product Categories by Revenue', fontsize=16, fontweight='bold')

# Add value labels
for bar, val in zip(bars, category_revenue['revenue']):
    ax.text(val + 1000, bar.get_y() + bar.get_height() / 2,
            f'RS {val:,.0f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('04_top_categories_revenue.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: 04_top_categories_revenue.png")

# Category Review Scores 
category_reviews = (
    df_delivered
    .groupby('product_category_name_english')
    .agg(
        avg_review=('review_score', 'mean'),
        total_orders=('order_id', 'nunique')
    )
    .query('total_orders >= 100')  # Only categories with 100+ orders
    .sort_values('avg_review', ascending=False)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Top 10 best rated
top10 = category_reviews.head(10)
axes[0].barh(top10['product_category_name_english'], top10['avg_review'],
             color='#4CAF50', alpha=0.8)
axes[0].invert_yaxis()
axes[0].set_xlabel('Average Review Score')
axes[0].set_title(' Top 10 Best Rated Categories', fontweight='bold')
axes[0].set_xlim(0, 5)

# Bottom 10 worst rated
bottom10 = category_reviews.tail(10)
axes[1].barh(bottom10['product_category_name_english'], bottom10['avg_review'],
             color='#F44336', alpha=0.8)
axes[1].invert_yaxis()
axes[1].set_xlabel('Average Review Score')
axes[1].set_title(' Bottom 10 Worst Rated Categories', fontweight='bold')
axes[1].set_xlim(0, 5)

plt.suptitle(' Product Category Satisfaction Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('05_category_reviews.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 05_category_reviews.png")

# CUSTOMER ANALYSIS
print("\n Generating Customer Analysis...")

# Customer Geography
customer_state = (
    df_delivered
    .groupby('customer_state')
    .agg(
        customers=('customer_unique_id', 'nunique'),
        revenue=('payment_value', 'sum'),
        orders=('order_id', 'nunique')
    )
    .sort_values('revenue', ascending=False)
    .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(customer_state['customer_state'], customer_state['revenue'],
              color=plt.cm.Blues(np.linspace(0.3, 1, len(customer_state))))
ax.set_xlabel('State')
ax.set_ylabel('Revenue (R$)')
ax.set_title(' Revenue by Customer State', fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('06_revenue_by_state.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 06_revenue_by_state.png")

# Repeat vs One-time Customers 
customer_orders = (
    df_delivered
    .groupby('customer_unique_id')['order_id']
    .nunique()
    .reset_index()
)
customer_orders.columns = ['customer_unique_id', 'total_orders']
customer_orders['customer_type'] = np.where(
    customer_orders['total_orders'] > 1, 'Repeat', 'One-time'
)

type_counts = customer_orders['customer_type'].value_counts()
repeat_rate = (type_counts.get('Repeat', 0) / type_counts.sum()) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
colors = ['#FF9800', '#4CAF50']
axes[0].pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Customer Type Distribution', fontweight='bold')

# Order frequency distribution
order_dist = customer_orders['total_orders'].value_counts().sort_index().head(10)
axes[1].bar(order_dist.index.astype(str), order_dist.values, color='#2196F3', alpha=0.8)
axes[1].set_xlabel('Number of Orders')
axes[1].set_ylabel('Number of Customers')
axes[1].set_title('Order Frequency Distribution', fontweight='bold')

plt.suptitle(f' Customer Loyalty (Repeat Rate: {repeat_rate:.1f}%)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('07_customer_loyalty.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"    Repeat Customer Rate: {repeat_rate:.1f}%")
print("    Saved: 07_customer_loyalty.png")

# RFM CUSTOMER SEGMENTATION
print("\n Performing RFM Segmentation...")

# Reference date (day after last order)
reference_date = df_delivered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

# Calculate RFM
rfm = (
    df_delivered
    .groupby('customer_unique_id')
    .agg(
        recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('payment_value', 'sum')
    )
    .reset_index()
)

# RFM Scoring (1-5 scale, 5 is best)
rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm['F_score'] = pd.cut(rfm['frequency'], bins=[0, 1, 2, 3, 5, np.inf],
                         labels=[1, 2, 3, 4, 5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Customer Segments
def segment_customer(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3 and m >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f >= 1 and m >= 2:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f <= 2 and m <= 2:
        return 'Lost'
    elif r <= 2 and f >= 2 and m >= 2:
        return 'Cant Lose Them'
    else:
        return 'Need Attention'

rfm['segment'] = rfm.apply(segment_customer, axis=1)

# Segment Summary
segment_summary = (
    rfm.groupby('segment')
    .agg(
        count=('customer_unique_id', 'count'),
        avg_recency=('recency', 'mean'),
        avg_frequency=('frequency', 'mean'),
        avg_monetary=('monetary', 'mean')
    )
    .sort_values('count', ascending=False)
    .reset_index()
)

print("\n   RFM Segment Summary:")
print("   " + "-" * 70)
for _, row in segment_summary.iterrows():
    print(f"   {row['segment']:20s} | Customers: {row['count']:6,} | "
          f"Avg Spend: R$ {row['avg_monetary']:8,.2f} | "
          f"Avg Recency: {row['avg_recency']:.0f} days")

# RFM Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Segment distribution
seg_counts = rfm['segment'].value_counts()
colors_rfm = plt.cm.Set3(np.linspace(0, 1, len(seg_counts)))
axes[0].barh(seg_counts.index, seg_counts.values, color=colors_rfm)
axes[0].set_xlabel('Number of Customers')
axes[0].set_title('Customer Segments', fontweight='bold')
axes[0].invert_yaxis()

# Recency vs Monetary scatter
scatter = axes[1].scatter(
    rfm['recency'], rfm['monetary'],
    c=rfm['R_score'], cmap='RdYlGn', alpha=0.5, s=10
)
axes[1].set_xlabel('Recency (days)')
axes[1].set_ylabel('Monetary (R$)')
axes[1].set_title('Recency vs Monetary', fontweight='bold')
plt.colorbar(scatter, ax=axes[1], label='R Score')

# Avg monetary by segment
seg_monetary = rfm.groupby('segment')['monetary'].mean().sort_values()
axes[2].barh(seg_monetary.index, seg_monetary.values, color='#FF9800', alpha=0.8)
axes[2].set_xlabel('Average Monetary Value (R$)')
axes[2].set_title('Avg Spend by Segment', fontweight='bold')

plt.suptitle(' RFM Customer Segmentation', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('08_rfm_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 08_rfm_segmentation.png")

# Save RFM data
rfm.to_csv('rfm_customer_segments.csv', index=False)
print("    Saved: rfm_customer_segments.csv")


# DELIVERY PERFORMANCE ANALYSIS
print("\n Analyzing Delivery Performance...")

# Filter records with delivery data
df_delivery = df_delivered.dropna(subset=['delivery_days', 'estimated_delivery_days']).copy()
df_delivery = df_delivery[df_delivery['delivery_days'] > 0]

avg_delivery = df_delivery['delivery_days'].mean()
median_delivery = df_delivery['delivery_days'].median()
on_time_rate = (df_delivery['delivery_status'] == 'On Time').mean() * 100

print(f"    Average Delivery Time:  {avg_delivery:.1f} days")
print(f"    Median Delivery Time:   {median_delivery:.1f} days")
print(f"    On-Time Delivery Rate:  {on_time_rate:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Delivery time distribution
axes[0, 0].hist(df_delivery['delivery_days'], bins=50, color='#2196F3',
                alpha=0.7, edgecolor='white')
axes[0, 0].axvline(avg_delivery, color='red', linestyle='--',
                    label=f'Mean: {avg_delivery:.1f} days')
axes[0, 0].axvline(median_delivery, color='green', linestyle='--',
                    label=f'Median: {median_delivery:.1f} days')
axes[0, 0].set_xlabel('Delivery Days')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Delivery Time Distribution', fontweight='bold')
axes[0, 0].legend()

# On-time vs Late
delivery_counts = df_delivery['delivery_status'].value_counts()
colors_del = ['#4CAF50', '#F44336']
axes[0, 1].pie(delivery_counts, labels=delivery_counts.index, autopct='%1.1f%%',
               colors=colors_del, startangle=90, textprops={'fontsize': 12})
axes[0, 1].set_title('On-Time vs Late Delivery', fontweight='bold')

# Delivery time by state (top 10)
state_delivery = (
    df_delivery
    .groupby('customer_state')['delivery_days']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
axes[1, 0].barh(state_delivery.index, state_delivery.values, color='#FF5722', alpha=0.8)
axes[1, 0].set_xlabel('Average Delivery Days')
axes[1, 0].set_title('Slowest Delivery States', fontweight='bold')
axes[1, 0].invert_yaxis()

# Review score vs delivery status
review_by_delivery = df_delivery.groupby('delivery_status')['review_score'].mean()
axes[1, 1].bar(review_by_delivery.index, review_by_delivery.values,
               color=['#4CAF50', '#F44336'], alpha=0.8)
axes[1, 1].set_ylabel('Average Review Score')
axes[1, 1].set_title('Review Score by Delivery Status', fontweight='bold')
axes[1, 1].set_ylim(0, 5)
for i, v in enumerate(review_by_delivery.values):
    axes[1, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')

plt.suptitle(' Delivery Performance Analysis', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('09_delivery_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 09_delivery_analysis.png")


# REVIEW SCORE ANALYSIS
print("\n Analyzing Customer Reviews...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Review distribution
review_dist = df_delivered['review_score'].value_counts().sort_index()
colors_rev = ['#F44336', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50']
axes[0].bar(review_dist.index, review_dist.values, color=colors_rev, alpha=0.8)
axes[0].set_xlabel('Review Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Review Score Distribution', fontweight='bold')
for i, v in zip(review_dist.index, review_dist.values):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

# Monthly average review
monthly_review = (
    df_delivered
    .groupby('order_year_month')['review_score']
    .mean()
    .reset_index()
)
monthly_review['order_year_month'] = monthly_review['order_year_month'].astype(str)
axes[1].plot(monthly_review['order_year_month'], monthly_review['review_score'],
             marker='o', color='#FF9800', linewidth=2)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Avg Review Score')
axes[1].set_title('Monthly Average Review Score Trend', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(0, 5)

plt.suptitle(' Customer Satisfaction Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('10_review_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: 10_review_analysis.png")


print("\n Analyzing Payment Methods...")

payment_analysis = (
    df_delivered
    .groupby('payment_type')
    .agg(
        count=('order_id', 'nunique'),
        total_value=('payment_value', 'sum'),
        avg_value=('payment_value', 'mean')
    )
    .sort_values('count', ascending=False)
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors_pay = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
axes[0].pie(payment_analysis['count'], labels=payment_analysis['payment_type'],
            autopct='%1.1f%%', colors=colors_pay[:len(payment_analysis)],
            startangle=90)
axes[0].set_title('Payment Method Distribution', fontweight='bold')

axes[1].bar(payment_analysis['payment_type'], payment_analysis['avg_value'],
            color=colors_pay[:len(payment_analysis)], alpha=0.8)
axes[1].set_ylabel('Average Payment Value (R$)')
axes[1].set_title('Avg Payment Value by Method', fontweight='bold')

plt.suptitle(' Payment Method Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('11_payment_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: 11_payment_analysis.png")


print("\n Correlation Analysis...")

corr_data = df_delivered[['price', 'freight_value', 'payment_value',
                          'review_score', 'delivery_days',
                          'product_weight_g', 'product_length_cm']].dropna()

corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=1, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('12_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: 12_correlation_heatmap.png")

print("\n Analyzing Seller Performance...")

seller_perf = (
    df_delivered
    .groupby('seller_id')
    .agg(
        total_revenue=('payment_value', 'sum'),
        total_orders=('order_id', 'nunique'),
        avg_review=('review_score', 'mean'),
        avg_delivery_days=('delivery_days', 'mean'),
        products_sold=('product_id', 'nunique')
    )
    .sort_values('total_revenue', ascending=False)
    .reset_index()
)

print(f"   Total Sellers: {seller_perf.shape[0]:,}")
print(f"   Top Seller Revenue: R$ {seller_perf['total_revenue'].iloc[0]:,.2f}")
print(f"   Top 10% sellers make: R$ {seller_perf.head(int(len(seller_perf)*0.1))['total_revenue'].sum():,.2f} "
      f"({seller_perf.head(int(len(seller_perf)*0.1))['total_revenue'].sum()/total_revenue*100:.1f}% of total)")

print("\n Month-over-Month Growth Analysis...")

monthly_revenue['revenue_growth'] = monthly_revenue['revenue'].pct_change() * 100
monthly_revenue['orders_growth'] = monthly_revenue['orders'].pct_change() * 100

fig, ax = plt.subplots(figsize=(14, 5))
colors_growth = ['#4CAF50' if x >= 0 else '#F44336' for x in monthly_revenue['revenue_growth'].fillna(0)]
ax.bar(monthly_revenue['order_year_month'], monthly_revenue['revenue_growth'],
       color=colors_growth, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('Month')
ax.set_ylabel('Growth Rate (%)')
ax.set_title(' Month-over-Month Revenue Growth', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('13_mom_growth.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved: 13_mom_growth.png")

print("\n Exporting Summary Data...")

# Save analysis summaries to Excel
with pd.ExcelWriter('ecommerce_analysis_summary.xlsx', engine='openpyxl') as writer:
    monthly_revenue.to_excel(writer, sheet_name='Monthly Revenue', index=False)
    category_revenue.to_excel(writer, sheet_name='Category Revenue', index=False)
    customer_state.to_excel(writer, sheet_name='State Revenue', index=False)
    segment_summary.to_excel(writer, sheet_name='RFM Segments', index=False)
    payment_analysis.to_excel(writer, sheet_name='Payment Methods', index=False)
    seller_perf.head(50).to_excel(writer, sheet_name='Top Sellers', index=False)

print(" Saved: ecommerce_analysis_summary.xlsx")

print("\n" + "=" * 60)
print(" ANALYSIS COMPLETE!")
print("=" * 60)
print(f"""
   FILES GENERATED:
   ├── 01_monthly_revenue_trend.png
   ├── 02_day_of_week_analysis.png
   ├── 03_hourly_orders.png
   ├── 04_top_categories_revenue.png
   ├── 05_category_reviews.png
   ├── 06_revenue_by_state.png
   ├── 07_customer_loyalty.png
   ├── 08_rfm_segmentation.png
   ├── 09_delivery_analysis.png
   ├── 10_review_analysis.png
   ├── 11_payment_analysis.png
   ├── 12_correlation_heatmap.png
   ├── 13_mom_growth.png
   ├── rfm_customer_segments.csv
   └── ecommerce_analysis_summary.xlsx

   KEY FINDINGS:
   1. Total Revenue: R$ {total_revenue:,.2f}
   2. Total Orders: {total_orders:,}
   3. Total Customers: {total_customers:,}
   4. Average Order Value: R$ {avg_order_value:,.2f}
   5. Average Review Score: {avg_review_score:.2f}/5.0
   6. On-Time Delivery Rate: {on_time_rate:.1f}%
   7. Repeat Customer Rate: {repeat_rate:.1f}%
""")