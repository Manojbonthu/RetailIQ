import os, sys, json, warnings, threading
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request
import joblib
from utils.chatbot import chat, set_data_context, get_quick_insights, save_api_key, get_saved_key

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'retailiq-secret-2024')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, 'models', 'master_cache.pkl')

DATA_PATHS = [
    os.path.join(BASE_DIR, 'data', 'Online_Retail.xlsx'),
    'data/Online_Retail.xlsx',
    'Online_Retail.xlsx',
]

_state = {
    'kpis': None, 'ml': None, 'context': {},
    'ready': False, 'error': None, 'status_msg': 'Starting...'
}

def find_data():
    for p in DATA_PATHS:
        if os.path.exists(p): return p
    return None

def _build_context(summary):
    ml = _state['ml']
    ctx = {
        'summary':          summary,
        'forecast_metrics': ml['forecast_metrics'],
        'segment_counts':   ml['segment_counts'],
        'top_products':     ml['top_products'],
        'revenue_forecast': ml['revenue_forecast'],
    }
    _state['context'] = ctx
    set_data_context(ctx)

# Bump this whenever you change data_processor or chatbot
# It auto-deletes old cache so data rebuilds fresh
CACHE_VERSION = 2

def boot():
    try:
        if os.path.exists(CACHE_FILE):
            _state['status_msg'] = 'Loading cached data...'
            print('[RetailIQ] Cache found — checking version...')
            master = joblib.load(CACHE_FILE)

            cached_summary = master.get('summary', '')
            cache_ver      = master.get('cache_version', 1)

            if cache_ver < CACHE_VERSION or 'MONTHLY REVENUE BREAKDOWN' not in cached_summary:
                print('[RetailIQ] Old cache — rebuilding with full data context...')
                os.remove(CACHE_FILE)
                # fall through to retrain
            else:
                _state['kpis'] = master['kpis']
                _state['ml']   = master['ml']
                _build_context(master['summary'])
                _state['ready'] = True
                print('[RetailIQ] Ready instantly from cache!')
                return

        print('[RetailIQ] First run — training models (one-time ~3 min)...')
        data_path = find_data()
        if not data_path:
            _state['error'] = 'Dataset not found. Place Online_Retail.xlsx in the data/ folder.'
            return

        from utils.data_processor import load_and_clean_data, compute_kpis, get_summary_stats
        from utils.ml_models import (train_revenue_forecast_model,
                                     train_customer_cluster_model,
                                     train_product_forecast)

        _state['status_msg'] = 'Loading transactions...'
        df = load_and_clean_data(data_path)
        print(f'[RetailIQ] Loaded {len(df):,} records')

        _state['status_msg'] = 'Computing KPIs...'
        kpis = compute_kpis(df)

        _state['status_msg'] = 'Training forecast model...'
        metrics, forecast, feat_imp, historical = train_revenue_forecast_model(df)

        _state['status_msg'] = 'Clustering customers...'
        seg_counts, cluster_stats, _ = train_customer_cluster_model(df)

        _state['status_msg'] = 'Analysing products...'
        top_products, product_trends = train_product_forecast(df)

        summary = get_summary_stats(df)
        ml = {
            'forecast_metrics': metrics,
            'revenue_forecast': forecast,
            'feature_importance': feat_imp,
            'historical_revenue': historical,
            'segment_counts':   seg_counts,
            'cluster_stats':    cluster_stats,
            'top_products':     top_products,
            'product_trends':   product_trends,
        }

        master = {
            'kpis': kpis,
            'ml':   ml,
            'summary': summary,
            'cache_version': CACHE_VERSION
        }
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
        joblib.dump(master, CACHE_FILE)
        print('[RetailIQ] Cache saved — future starts will be instant!')

        _state['kpis'] = kpis
        _state['ml']   = ml
        _build_context(summary)
        _state['ready'] = True
        print('[RetailIQ] System ready!')

    except Exception as e:
        import traceback
        _state['error'] = str(e)
        print(f'[RetailIQ] Boot error: {e}')
        traceback.print_exc()

threading.Thread(target=boot, daemon=True).start()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/status')
def status():
    return jsonify({
        'ready':   _state['ready'],
        'error':   _state['error'],
        'message': _state['status_msg'],
        'cached':  os.path.exists(CACHE_FILE)
    })

@app.route('/api/kpis')
def get_kpis():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    k = _state['kpis']
    return jsonify({
        'total_revenue':   k['total_revenue'],
        'total_orders':    k['total_orders'],
        'total_customers': k['total_customers'],
        'total_products':  k['total_products'],
        'avg_order_value': k['avg_order_value'],
        'top_country':     k['top_country'],
    })

@app.route('/api/charts/revenue-trend')
def revenue_trend():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['monthly_revenue'])

@app.route('/api/charts/top-products')
def top_products():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['top_products'])

@app.route('/api/charts/country-revenue')
def country_revenue():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['country_revenue'])

@app.route('/api/charts/rfm-segments')
def rfm_segments():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['rfm_segments'])

@app.route('/api/charts/hourly-pattern')
def hourly_pattern():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['hourly_pattern'])

@app.route('/api/charts/dow-pattern')
def dow_pattern():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify(_state['kpis']['dow_pattern'])

@app.route('/api/ml/forecast')
def ml_forecast():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    ml = _state['ml']
    return jsonify({
        'metrics':            ml['forecast_metrics'],
        'forecast':           ml['revenue_forecast'],
        'historical':         ml['historical_revenue'],
        'feature_importance': ml['feature_importance']
    })

@app.route('/api/ml/segments')
def ml_segments():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    ml = _state['ml']
    return jsonify({
        'segment_counts': ml['segment_counts'],
        'cluster_stats':  ml['cluster_stats']
    })

@app.route('/api/ml/products')
def ml_products():
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    ml = _state['ml']
    return jsonify({
        'top_products':   ml['top_products'],
        'product_trends': ml['product_trends']
    })

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message: return jsonify({'error': 'empty'}), 400
    return jsonify({
        'response': chat(
            message,
            data.get('history', []),
            data.get('api_key', '').strip()
        )
    })

@app.route('/api/insights', methods=['POST'])
def auto_insights():
    data = request.get_json()
    if not _state['ready']: return jsonify({'error': 'loading'}), 503
    return jsonify({
        'insights': get_quick_insights(
            data.get('api_key', '').strip(),
            _state['context']
        )
    })

@app.route('/api/save-key', methods=['POST'])
def save_key():
    data = request.get_json()
    key  = data.get('api_key', '').strip()
    if not key:
        return jsonify({'ok': False, 'message': 'No key provided'}), 400
    save_api_key(key)
    return jsonify({'ok': True, 'message': 'Groq API key saved.'})

@app.route('/api/key-status')
def key_status():
    k = get_saved_key()
    return jsonify({
        'has_key': bool(k),
        'preview': k[:12] + '...' if k else ''
    })

@app.route('/api/cache/status')
def cache_status():
    exists = os.path.exists(CACHE_FILE)
    size   = round(os.path.getsize(CACHE_FILE) / 1024, 1) if exists else 0
    return jsonify({'exists': exists, 'size_kb': size})

@app.route('/api/cache/clear', methods=['POST'])
def cache_clear():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
        return jsonify({'message': 'Cache cleared. Restart to retrain.'})
    return jsonify({'message': 'No cache found.'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
