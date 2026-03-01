import pandas as pd
import numpy as np
import joblib
import os
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def train_revenue_forecast_model(df):
    """Train a lightweight GBM model to forecast daily revenue."""
    # Aggregate to daily — much smaller than raw data
    daily = df.groupby(df['InvoiceDate'].dt.date).agg(
        Revenue=('Revenue', 'sum'),
        NumOrders=('InvoiceNo', 'nunique'),
        NumCustomers=('CustomerID', 'nunique'),
        NumItems=('Quantity', 'sum')
    ).reset_index()

    daily['InvoiceDate'] = pd.to_datetime(daily['InvoiceDate'])
    daily = daily.sort_values('InvoiceDate').reset_index(drop=True)

    # Feature engineering
    daily['DayOfWeek']  = daily['InvoiceDate'].dt.dayofweek
    daily['Month']      = daily['InvoiceDate'].dt.month
    daily['DayOfMonth'] = daily['InvoiceDate'].dt.day
    daily['WeekOfYear'] = daily['InvoiceDate'].dt.isocalendar().week.astype(int)
    daily['DayOfYear']  = daily['InvoiceDate'].dt.dayofyear

    for lag in [1, 7, 14, 30]:
        daily[f'Revenue_lag{lag}'] = daily['Revenue'].shift(lag)

    daily['Revenue_roll7']     = daily['Revenue'].shift(1).rolling(7).mean()
    daily['Revenue_roll14']    = daily['Revenue'].shift(1).rolling(14).mean()
    daily['Revenue_roll30']    = daily['Revenue'].shift(1).rolling(30).mean()
    daily['Revenue_roll7_std'] = daily['Revenue'].shift(1).rolling(7).std()

    daily = daily.dropna()

    features = [
        'DayOfWeek', 'Month', 'DayOfMonth', 'WeekOfYear', 'DayOfYear',
        'NumOrders', 'NumCustomers', 'NumItems',
        'Revenue_lag1', 'Revenue_lag7', 'Revenue_lag14', 'Revenue_lag30',
        'Revenue_roll7', 'Revenue_roll14', 'Revenue_roll30', 'Revenue_roll7_std'
    ]

    X = daily[features]
    y = daily['Revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ── Kept small for Render free tier (512 MB RAM) ──────────────────────────
    model = GradientBoostingRegressor(
        n_estimators=50,        # reduced from 100 → faster + less RAM
        learning_rate=0.08,
        max_depth=4,            # shallower trees → less RAM
        min_samples_split=15,
        subsample=0.8,
        max_features=0.8,       # use 80% of features per split → faster
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'mae':  round(float(mean_absolute_error(y_test, y_pred)), 2),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        'r2':   round(float(r2_score(y_test, y_pred)), 4),
        'mape': round(float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100), 2),
    }

    feat_imp = pd.Series(model.feature_importances_, index=features).nlargest(10)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {'model': model, 'features': features},
        os.path.join(MODELS_DIR, 'revenue_forecast.pkl'),
        compress=3,
    )

    # 30-day rolling forecast
    forecast = []
    last_date       = daily['InvoiceDate'].iloc[-1]
    recent_revenues = list(daily['Revenue'].values[-30:])
    avg_orders      = float(daily['NumOrders'].mean())
    avg_customers   = float(daily['NumCustomers'].mean())
    avg_items       = float(daily['NumItems'].mean())

    for i in range(1, 31):
        next_date = last_date + pd.Timedelta(days=i)
        row = {
            'DayOfWeek':        next_date.dayofweek,
            'Month':            next_date.month,
            'DayOfMonth':       next_date.day,
            'WeekOfYear':       next_date.isocalendar()[1],
            'DayOfYear':        next_date.dayofyear,
            'NumOrders':        avg_orders,
            'NumCustomers':     avg_customers,
            'NumItems':         avg_items,
            'Revenue_lag1':     recent_revenues[-1]  if len(recent_revenues) >= 1  else 0,
            'Revenue_lag7':     recent_revenues[-7]  if len(recent_revenues) >= 7  else 0,
            'Revenue_lag14':    recent_revenues[-14] if len(recent_revenues) >= 14 else 0,
            'Revenue_lag30':    recent_revenues[-30] if len(recent_revenues) >= 30 else 0,
            'Revenue_roll7':    np.mean(recent_revenues[-7:])  if len(recent_revenues) >= 7  else np.mean(recent_revenues),
            'Revenue_roll14':   np.mean(recent_revenues[-14:]) if len(recent_revenues) >= 14 else np.mean(recent_revenues),
            'Revenue_roll30':   np.mean(recent_revenues[-30:]) if len(recent_revenues) >= 30 else np.mean(recent_revenues),
            'Revenue_roll7_std': np.std(recent_revenues[-7:])  if len(recent_revenues) >= 7  else np.std(recent_revenues),
        }
        X_pred = pd.DataFrame([row])[features]
        pred   = max(0.0, float(model.predict(X_pred)[0]))
        forecast.append({'Date': next_date.strftime('%Y-%m-%d'), 'Predicted_Revenue': round(pred, 2)})
        recent_revenues.append(pred)
        if len(recent_revenues) > 30:
            recent_revenues.pop(0)

    historical = (
        daily[['InvoiceDate', 'Revenue']]
        .tail(60)
        .assign(InvoiceDate=lambda x: x['InvoiceDate'].dt.strftime('%Y-%m-%d'))
        .to_dict('records')
    )

    # Free memory
    del daily, X, y, X_train, X_test, y_train, y_test
    gc.collect()

    return metrics, forecast, feat_imp.to_dict(), historical


def train_customer_cluster_model(df):
    """Customer segmentation using K-Means on RFM features."""
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo',  'nunique'),
        Monetary=('Revenue',     'sum')
    ).reset_index()

    scaler     = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans         = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = (
        rfm.groupby('Cluster')
        .agg(Recency=('Recency', 'mean'), Frequency=('Frequency', 'mean'),
             Monetary=('Monetary', 'mean'), Count=('CustomerID', 'count'))
        .reset_index()
        .sort_values('Monetary', ascending=False)
        .reset_index(drop=True)
    )

    labels = {
        cluster_summary.iloc[0]['Cluster']: 'Champions',
        cluster_summary.iloc[1]['Cluster']: 'Loyal',
        cluster_summary.iloc[2]['Cluster']: 'Potential',
        cluster_summary.iloc[3]['Cluster']: 'At Risk',
    }
    rfm['SegmentLabel'] = rfm['Cluster'].map(labels)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {'model': kmeans, 'scaler': scaler, 'labels': labels},
        os.path.join(MODELS_DIR, 'customer_cluster.pkl'),
        compress=3,
    )

    segment_counts         = rfm['SegmentLabel'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    cluster_stats = (
        rfm.groupby('SegmentLabel')
        .agg(Avg_Recency=('Recency', 'mean'), Avg_Frequency=('Frequency', 'mean'),
             Avg_Monetary=('Monetary', 'mean'), Count=('CustomerID', 'count'))
        .reset_index()
        .round(2)
    )

    result_rfm = rfm.copy()

    del rfm, rfm_scaled
    gc.collect()

    return segment_counts.to_dict('records'), cluster_stats.to_dict('records'), result_rfm


def train_product_forecast(df):
    """Product-level revenue ranking and monthly trend."""
    product_monthly = df.groupby(
        ['Description', 'Year', 'Month']
    )['Revenue'].sum().reset_index()

    top_products = (
        df.groupby('Description')['Revenue']
        .sum()
        .nlargest(20)
        .reset_index()
    )

    product_trends = []
    for prod in top_products['Description'].values[:5]:
        trend = product_monthly[product_monthly['Description'] == prod].copy()
        trend['Period'] = (
            trend['Year'].astype(str) + '-' +
            trend['Month'].astype(str).str.zfill(2)
        )
        product_trends.append({
            'product': prod,
            'trend':   trend[['Period', 'Revenue']].to_dict('records'),
        })

    del product_monthly
    gc.collect()

    return top_products.to_dict('records'), product_trends
