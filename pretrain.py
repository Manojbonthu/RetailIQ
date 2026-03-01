"""
pretrain.py — Run during Render BUILD step (not runtime).

This trains all ML models and saves master_cache.pkl so the web
server starts instantly without needing to train on 512MB free RAM.

Called by render.yaml buildCommand BEFORE gunicorn starts.
"""

import os, sys, gc, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'models', 'master_cache.pkl')
CACHE_VERSION = 3

DATA_PATHS = [
    os.path.join(BASE_DIR, 'data', 'Online_Retail.xlsx'),
    'data/Online_Retail.xlsx',
    'Online_Retail.xlsx',
]

def find_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            return p
    return None


def main():
    print('=' * 60)
    print('[pretrain] Starting pre-training build step...')
    print('=' * 60)

    # Skip if fresh valid cache already exists
    if os.path.exists(CACHE_FILE):
        try:
            master = joblib.load(CACHE_FILE)
            if (master.get('cache_version', 0) >= CACHE_VERSION and
                    'MONTHLY REVENUE BREAKDOWN' in master.get('summary', '')):
                print('[pretrain] Valid cache already exists — skipping training.')
                return
        except Exception:
            pass
        os.remove(CACHE_FILE)

    data_path = find_data()
    if not data_path:
        print('[pretrain] WARNING: data/Online_Retail.xlsx not found.')
        print('[pretrain] The app will attempt training at runtime instead.')
        return

    print(f'[pretrain] Dataset found: {data_path}')

    from utils.data_processor import load_and_clean_data, compute_kpis, get_summary_stats
    from utils.ml_models import (train_revenue_forecast_model,
                                 train_customer_cluster_model,
                                 train_product_forecast)

    print('[pretrain] Loading & cleaning data...')
    df = load_and_clean_data(data_path)
    print(f'[pretrain] Loaded {len(df):,} records')
    gc.collect()

    print('[pretrain] Computing KPIs...')
    kpis = compute_kpis(df)
    gc.collect()

    print('[pretrain] Training revenue forecast model...')
    metrics, forecast, feat_imp, historical = train_revenue_forecast_model(df)
    gc.collect()

    print('[pretrain] Clustering customers...')
    seg_counts, cluster_stats, _ = train_customer_cluster_model(df)
    gc.collect()

    print('[pretrain] Analysing products...')
    top_products, product_trends = train_product_forecast(df)

    print('[pretrain] Building summary stats...')
    summary = get_summary_stats(df)

    del df
    gc.collect()

    ml = {
        'forecast_metrics':  metrics,
        'revenue_forecast':  forecast,
        'feature_importance': feat_imp,
        'historical_revenue': historical,
        'segment_counts':    seg_counts,
        'cluster_stats':     cluster_stats,
        'top_products':      top_products,
        'product_trends':    product_trends,
    }

    master = {
        'kpis':          kpis,
        'ml':            ml,
        'summary':       summary,
        'cache_version': CACHE_VERSION,
    }

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
    joblib.dump(master, CACHE_FILE, compress=3)

    size_mb = os.path.getsize(CACHE_FILE) / 1024 / 1024
    print(f'[pretrain] Cache saved ({size_mb:.1f} MB) → {CACHE_FILE}')
    print('[pretrain] Pre-training complete! App will start instantly.')
    print('=' * 60)


if __name__ == '__main__':
    main()
