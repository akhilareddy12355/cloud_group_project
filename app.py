from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, Response, stream_with_context
)
import pyodbc
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

# --- Configure upload folders if needed ---
app.config['Upload_folder_HouseHolds']    = os.getenv('UPLOAD_FOLDER_HOUSEHOLDS', './uploads/households')
app.config['Upload_folder_Products']      = os.getenv('UPLOAD_FOLDER_PRODUCTS', './uploads/products')
app.config['Upload_folder_Transactions']  = os.getenv('UPLOAD_FOLDER_TRANSACTIONS', './uploads/transactions')

# --- Helper stubs for file uploads (implement as needed) ---
ALLOWED_EXTENSIONS = {'csv'}
def check_file_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def readCSVandloaddata(filepath, table_name):
    # TODO: implement reading the CSV and inserting into your database
    pass

# --- SQLAlchemy engine for pandas reads ---
server   = 'kogerserver2025.database.windows.net'
database = 'krogerdb'
username = 'akhila'
password = urllib.parse.quote_plus('Reddy1234')
connection_string = (
    f"mssql+pyodbc://{username}:{password}@{server}/{database}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
engine = create_engine(connection_string, pool_pre_ping=True)

# --- Fallback pyodbc connection for writes, raw queries, etc. ---
def get_db_connection():
    conn_str = (
        'DRIVER={ODBC Driver 17 for SQL Server};'
        f"SERVER={server};DATABASE={database};"
        f"UID={username};PWD=Reddy1234"
    )
    return pyodbc.connect(conn_str)

# --- Streaming endpoint to avoid OOM on large payloads ---
@app.route('/demographicsandengagement', methods=['GET'])
def demographicsandengagement():
    query = """
        SELECT hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN, SUM(tr.SPEND) AS TOTAL_SPEND
        FROM transactions tr
        INNER JOIN households hh ON tr.HSHD_NUM = hh.HSHD_NUM
        GROUP BY hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN
    """

    def generate():
        yield "["               # start JSON array
        first = True
        for chunk_df in pd.read_sql_query(query, engine, chunksize=2000):
            records = chunk_df.to_dict(orient="records")
            chunk_json = json.dumps(records)[1:-1]  # strip outer [ ]
            if not first:
                yield ","
            yield chunk_json
            first = False
        yield "]"               # close JSON array

    return Response(
        stream_with_context(generate()),
        mimetype="application/json"
    )

# --- Home & redirect ---
@app.route('/')
def index():
    return redirect(url_for('signup'))

@app.route('/home')
def home():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT HSHD_NUM FROM households")
    hshd_nums = cur.fetchall()
    cur.close()
    conn.close()
    return render_template('home.html', hshd_nums=hshd_nums)

# --- Dashboard data fetch ---
@app.route('/get_dashboard_data', methods=['GET'])
def get_dashboard_data():
    hshd_num = request.args.get('hshd_num')
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        SELECT h.HSHD_NUM, t.BASKET_NUM, t.PURCHASE, t.PRODUCT_NUM, p.DEPARTMENT,
               p.COMMODITY, t.SPEND, t.UNITS, t.STORE_R, t.WEEK_NUM, t.YEAR,
               h.L, h.AGE_RANGE, h.MARITAL, h.INCOME_RANGE, h.HOMEOWNER,
               h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
        FROM transactions t
        JOIN households h ON t.HSHD_NUM = h.HSHD_NUM
        JOIN products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
        WHERE h.HSHD_NUM = ?
        ORDER BY t.YEAR, t.WEEK_NUM, t.PRODUCT_NUM
    """
    cur.execute(query, (hshd_num,))
    rows = cur.fetchall()
    columns = [
        'HSHD_NUM','BASKET_NUM','PURCHASE','PRODUCT_NUM','DEPARTMENT',
        'COMMODITY','SPEND','UNITS','STORE_R','WEEK_NUM','YEAR','L',
        'AGE_RANGE','MARITAL','INCOME_RANGE','HOMEOWNER',
        'HSHD_COMPOSITION','HH_SIZE','CHILDREN'
    ]
    data = []
    for row in rows:
        cleaned = {}
        for i, v in enumerate(row):
            if v is None or (isinstance(v, str) and v.strip().lower() in ['', 'null']):
                cleaned[columns[i]] = "N/A"
            else:
                cleaned[columns[i]] = v
        data.append(cleaned)
    cur.close()
    conn.close()
    return jsonify(data)

# --- Data loading & feature prep for modeling endpoints ---
def load_data():
    try:
        conn = get_db_connection()
        hh_q = "SELECT * FROM households"
        pr_q = "SELECT * FROM products"
        tr_q = "SELECT * FROM transactions"
        households   = pd.read_sql(hh_q, conn)
        products     = pd.read_sql(pr_q, conn)
        transactions = pd.read_sql(tr_q, conn)
        conn.close()
        return households, products, transactions
    except Exception as e:
        print(f"Database error: {e}")
        return None, None, None

def prepare_features(hh, pr, tr):
    # Clean numeric columns
    for col in ['HH_SIZE','CHILDREN']:
        hh[col] = (hh[col].astype(str).str.strip()
                    .replace('null', pd.NA)
                    .pipe(pd.to_numeric, errors='coerce').fillna(0))
    for col in ['SPEND','UNITS','WEEK_NUM','YEAR']:
        tr[col] = (tr[col].astype(str).str.strip()
                    .replace('null', pd.NA)
                    .pipe(pd.to_numeric, errors='coerce').fillna(0))
    merged = tr.merge(hh, on='HSHD_NUM', how='left').merge(pr, on='PRODUCT_NUM', how='left')
    cats = [
        'AGE_RANGE','MARITAL','INCOME_RANGE','HOMEOWNER','HSHD_COMPOSITION',
        'DEPARTMENT','COMMODITY','BRAND_TY','NATURAL_ORGANIC_FLAG','STORE_R'
    ]
    for c in cats:
        merged[c] = merged[c].astype(str).str.strip().replace('null', merged[c].mode()[0])
    le = LabelEncoder()
    for c in cats:
        merged[c + '_encoded'] = le.fit_transform(merged[c])
    features = [c + '_encoded' for c in cats] + ['HH_SIZE','CHILDREN','WEEK_NUM','YEAR']
    X = merged[features]
    y = merged['SPEND']
    return X, y, merged

# --- Model training endpoint ---
@app.route('/train_model')
def train_model():
    hh, pr, tr = load_data()
    if hh is None:
        return jsonify({'error':'DB load failed'}), 500
    X, y, merged = prepare_features(hh, pr, tr)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f'gb_model_{ts}.pkl'
    joblib.dump(model, fname)
    # Persist metrics
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        IF NOT EXISTS (
          SELECT * FROM sysobjects WHERE name='model_metrics' AND xtype='U'
        )
        CREATE TABLE model_metrics (
          id INT IDENTITY(1,1) PRIMARY KEY,
          model_filename VARCHAR(255),
          mse FLOAT, timestamp DATETIME
        )
    """)
    cur.execute(
        "INSERT INTO model_metrics (model_filename,mse,timestamp) VALUES (?,?,GETDATE())",
        (fname, mse)
    )
    conn.commit(); conn.close()
    return jsonify({'mse': mse, 'model_filename': fname})

# --- Model status endpoint ---
@app.route('/get_model_status')
def get_model_status():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT TOP 1 model_filename, mse, timestamp
        FROM model_metrics
        ORDER BY timestamp DESC
    """)
    row = cur.fetchone()
    conn.close()
    if row:
        return jsonify({
            'exists': True,
            'model_filename': row[0],
            'mse': row[1],
            'last_trained': row[2].strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify({'exists': False})

# --- Prediction endpoint ---
@app.route('/predict')
def predict():
    model = joblib.load('gb_model.pkl')
    hh, pr, tr = load_data()
    X, _, merged = prepare_features(hh, pr, tr)
    preds = model.predict(X)
    merged['predicted_spend'] = preds
    top = (
        merged.groupby(['DEPARTMENT','COMMODITY'])
        .agg({'predicted_spend':'mean','PRODUCT_NUM':'count'})
        .sort_values('predicted_spend', ascending=False)
        .head(10)
    )
    return jsonify({
        'department_commodity': top.index.tolist(),
        'predicted_spend': top['predicted_spend'].round(2).tolist(),
        'product_count': top['PRODUCT_NUM'].tolist()
    })

# --- Analytics endpoints ---
@app.route('/get_analytics')
def get_analytics():
    hh, pr, tr = load_data()
    X, y, merged = prepare_features(hh, pr, tr)
    features = [
        'login_frequency','session_duration','interaction_count',
        'purchase_value','support_tickets','email_response_rate','customer_status'
    ]
    for feature in features:
        if feature not in merged.columns:
            if feature == 'customer_status':
                merged[feature] = np.random.choice([0, 1], size=len(merged))
            else:
                merged[feature] = np.random.random(size=len(merged))
    corr = merged[features].corr()
    importance = corr['customer_status'].sort_values(ascending=True)
    return jsonify({
        'correlation_matrix': {
            'z': corr.values.tolist(),
            'x': corr.columns.tolist(),
            'y': corr.columns.tolist(),
            'text': corr.round(2).values.tolist()
        },
        'feature_importance': {
            'features': importance.index.tolist(),
            'importance': importance.values.tolist()
        }
    })

@app.route('/analyze_transactions')
def analyze_transactions():
    hh, pr, tr = load_data()
    merged = tr.merge(hh, on='HSHD_NUM', how='left').merge(pr, on='PRODUCT_NUM', how='left')
    metrics = {
        'transaction_frequency': merged.groupby('HSHD_NUM')['BASKET_NUM'].count(),
        'average_spend': merged.groupby('HSHD_NUM')['SPEND'].mean(),
        'product_diversity': merged.groupby('HSHD_NUM')['COMMODITY'].nunique()
    }
    engagement = pd.DataFrame(metrics)
    engagement['engagement_score'] = (
        (engagement['transaction_frequency'] / engagement['transaction_frequency'].max()) +
        (engagement['average_spend'] / engagement['average_spend'].max()) +
        (engagement['product_diversity'] / engagement['product_diversity'].max())
    ) / 3
    risk_threshold = engagement['engagement_score'].quantile(0.2)
    at_risk = engagement[engagement['engagement_score'] < risk_threshold]
    return jsonify({
        'engagement_metrics': {
            'scores': engagement['engagement_score'].tolist(),
            'households': engagement.index.tolist(),
            'risk_threshold': float(risk_threshold)
        },
        'at_risk_count': len(at_risk),
        'total_customers': len(engagement)
    })

@app.route('/retention_analysis')
def retention_analysis():
    hh, pr, tr = load_data()
    tr['PURCHASE_DATE'] = pd.to_datetime(
        tr['YEAR'].astype(str) + '-W' +
        tr['WEEK_NUM'].astype(str).str.zfill(2) + '-1',
        format='%Y-W%W-%w'
    )
    purchase_gaps = tr.groupby('HSHD_NUM')['PURCHASE_DATE'].agg(['min', 'max'])
    purchase_gaps['days_active'] = (purchase_gaps['max'] - purchase_gaps['min']).dt.days
    customer_value = tr.groupby('HSHD_NUM').agg({
        'SPEND': 'sum',
        'BASKET_NUM': 'nunique'
    }).fillna(0)
    return jsonify({
        'customer_lifetime': purchase_gaps['days_active'].tolist(),
        'customer_value': customer_value['SPEND'].tolist(),
        'basket_count': customer_value['BASKET_NUM'].tolist()
    })

# --- User signup/login/logout ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if not username or not password or not email:
            return "All fields are required!", 400
        conn = get_db_connection(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO Users (Username, Password, Email) VALUES (?,?,?)",
            (username, password, email)
        )
        conn.commit(); cur.close(); conn.close()
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/predictproducts')
def predictproducts():
    return render_template('predict.html')

@app.route('/logout')
def logout():
    # Clear session if used
    return redirect(url_for('signup'))

# --- Data upload endpoints ---
@app.route('/uploaddatasets', methods=['GET','POST'])
def uploaddatasets():
    return render_template('UploadData.html')

@app.route('/storeuploadedhouseholdfile', methods=['GET','POST'])
def storeuploadedhouseholdfile():
    message = 'Please upload file again!!'
    if request.method == 'POST':
        f = request.files['file']
        if check_file_extension(f.filename):
            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config['Upload_folder_HouseHolds'], filename)
            f.save(save_path)
            readCSVandloaddata(save_path, "households")
            message = 'File uploaded successfully'
        else:
            message = 'The file extension is not allowed'
    return render_template('UploadData.html', message=message)

@app.route('/storeuploadedProductfile', methods=['GET','POST'])
def storeuploadedProductfile():
    messageProducts = 'Please upload file again!!'
    if request.method == 'POST':
        f = request.files['file']
        if check_file_extension(f.filename):
            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config['Upload_folder_Products'], filename)
            f.save(save_path)
            readCSVandloaddata(save_path, "products")
            messageProducts = 'File uploaded successfully'
        else:
            messageProducts = 'The file extension is not allowed'
    return render_template('UploadData.html', messageProducts=messageProducts)

@app.route('/storeuploadedTransactionfile', methods=['GET','POST'])
def storeuploadedTransactionfile():
    messageTransactions = 'Please upload file again!!'
    if request.method == 'POST':
        f = request.files['file']
        if check_file_extension(f.filename):
            filename = secure_filename(f.filename)
            save_path = os.path.join(app.config['Upload_folder_Transactions'], filename)
            f.save(save_path)
            readCSVandloaddata(save_path, "transactions")
            messageTransactions = 'File uploaded successfully'
        else:
            messageTransactions = 'The file extension is not allowed'
    return render_template('UploadData.html', messageTransactions=messageTransactions)

if __name__ == '__main__':
    app.run(debug=True)
