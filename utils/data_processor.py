import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath):
    """Load and clean the retail dataset."""
    df = pd.read_excel(filepath)

    # Remove cancelled orders (InvoiceNo starting with C)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Remove rows with null CustomerID
    df = df.dropna(subset=['CustomerID'])

    # Remove invalid quantities and prices
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]

    # Fill description nulls
    df['Description'] = df['Description'].fillna('Unknown')

    # Create Revenue column
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    # Date features
    df['Year']      = df['InvoiceDate'].dt.year
    df['Month']     = df['InvoiceDate'].dt.month
    df['Day']       = df['InvoiceDate'].dt.day
    df['Hour']      = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['WeekOfYear']= df['InvoiceDate'].dt.isocalendar().week.astype(int)

    df['CustomerID'] = df['CustomerID'].astype(int)

    return df


def compute_kpis(df):
    """Compute key business KPIs."""
    kpis = {}

    kpis['total_revenue']   = round(df['Revenue'].sum(), 2)
    kpis['total_orders']    = df['InvoiceNo'].nunique()
    kpis['total_customers'] = df['CustomerID'].nunique()
    kpis['total_products']  = df['StockCode'].nunique()
    kpis['avg_order_value'] = round(df.groupby('InvoiceNo')['Revenue'].sum().mean(), 2)
    kpis['top_country']     = df.groupby('Country')['Revenue'].sum().idxmax()

    # Monthly revenue trend
    monthly = df.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
    monthly['Period'] = monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str).str.zfill(2)
    kpis['monthly_revenue'] = monthly[['Period', 'Revenue']].to_dict('records')

    # Top 10 products by revenue
    top_products = df.groupby('Description')['Revenue'].sum().nlargest(10).reset_index()
    kpis['top_products'] = top_products.to_dict('records')

    # Revenue by country
    country_rev = df.groupby('Country')['Revenue'].sum().nlargest(10).reset_index()
    kpis['country_revenue'] = country_rev.to_dict('records')

    # Daily revenue for recent period
    daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    daily.columns = ['Date', 'Revenue']
    daily['Date'] = daily['Date'].astype(str)
    kpis['daily_revenue'] = daily.tail(60).to_dict('records')

    # Customer segmentation via RFM
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo',  'nunique'),
        Monetary=('Revenue',     'sum')
    ).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'),  4, labels=[1,2,3,4]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    def segment(score):
        if score >= 10: return 'Champions'
        elif score >= 7: return 'Loyal Customers'
        elif score >= 5: return 'At Risk'
        else:            return 'Lost Customers'

    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    kpis['rfm_segments'] = (
        rfm['Segment'].value_counts()
        .reset_index()
        .rename(columns={'index': 'Segment', 'count': 'Count'})
        .to_dict('records')
    )
    kpis['rfm_data'] = rfm[['CustomerID','Recency','Frequency','Monetary','Segment']].head(100).to_dict('records')

    # Hourly sales pattern
    hourly = df.groupby('Hour')['Revenue'].sum().reset_index()
    kpis['hourly_pattern'] = hourly.to_dict('records')

    # Day of week pattern
    dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    dow = df.groupby('DayOfWeek')['Revenue'].sum().reset_index()
    dow['DayName'] = dow['DayOfWeek'].map(dow_map)
    kpis['dow_pattern'] = dow[['DayName','Revenue']].to_dict('records')

    return kpis


def get_summary_stats(df):
    """Build a RICH text summary — every metric the chatbot needs to answer questions."""
    kpis = compute_kpis(df)

    # Monthly breakdown — full history
    monthly_lines = '\n'.join(
        f"  {r['Period']}: £{r['Revenue']:,.0f}"
        for r in kpis['monthly_revenue']
    )

    # Top 10 products
    product_lines = '\n'.join(
        f"  {i+1}. {p['Description']}: £{p['Revenue']:,.0f}"
        for i, p in enumerate(kpis['top_products'][:10])
    )

    # Top 10 countries
    country_lines = '\n'.join(
        f"  {i+1}. {c['Country']}: £{c['Revenue']:,.0f}"
        for i, c in enumerate(kpis['country_revenue'][:10])
    )

    # Customer segments
    segment_lines = '\n'.join(
        f"  {s['Segment']}: {s['Count']} customers"
        for s in kpis['rfm_segments']
    )

    # Hourly pattern (compact)
    hourly_lines = ', '.join(
        f"{int(h['Hour'])}h=£{h['Revenue']:,.0f}"
        for h in kpis['hourly_pattern']
    )

    # Day of week pattern
    dow_lines = ', '.join(
        f"{d['DayName']}=£{d['Revenue']:,.0f}"
        for d in kpis['dow_pattern']
    )

    # Best / worst month helpers
    best_month  = max(kpis['monthly_revenue'], key=lambda x: x['Revenue'])
    worst_month = min(kpis['monthly_revenue'], key=lambda x: x['Revenue'])

    months_2011 = sorted(r['Period'] for r in kpis['monthly_revenue'] if r['Period'].startswith('2011'))
    months_2010 = sorted(r['Period'] for r in kpis['monthly_revenue'] if r['Period'].startswith('2010'))

    summary = (
        "RETAIL BUSINESS DATASET - FULL CONTEXT\n"
        "=======================================\n"
        f"Date Range   : Dec 2010 to Dec 2011\n"
        f"Business     : UK-based online retail, unique all-occasion gifts\n"
        f"Total Revenue: £{kpis['total_revenue']:,.2f}\n"
        f"Total Orders : {kpis['total_orders']:,}\n"
        f"Customers    : {kpis['total_customers']:,} unique\n"
        f"Products     : {kpis['total_products']:,} unique SKUs\n"
        f"Avg Order Val: £{kpis['avg_order_value']:,.2f}\n"
        f"Top Country  : {kpis['top_country']}\n\n"

        "MONTHLY REVENUE BREAKDOWN (complete history):\n"
        f"{monthly_lines}\n\n"

        "KEY MONTHLY FACTS:\n"
        f"- Highest revenue month : {best_month['Period']} (£{best_month['Revenue']:,.0f})\n"
        f"- Lowest revenue month  : {worst_month['Period']} (£{worst_month['Revenue']:,.0f})\n"
        f"- Available 2011 months : {', '.join(months_2011)}\n"
        f"- Available 2010 months : {', '.join(months_2010)}\n\n"

        "TOP 10 PRODUCTS BY REVENUE:\n"
        f"{product_lines}\n\n"

        "TOP 10 COUNTRIES BY REVENUE:\n"
        f"{country_lines}\n\n"

        "CUSTOMER SEGMENTS (RFM Analysis):\n"
        f"{segment_lines}\n\n"

        "SALES BY HOUR OF DAY:\n"
        f"{hourly_lines}\n\n"

        "SALES BY DAY OF WEEK:\n"
        f"{dow_lines}\n"
    )
    return summary.strip()
