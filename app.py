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
import glob
import os


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
# --- Dashboard: Demographics & Engagement ---
@app.route('/demographicsandengagement')
def demographicsandengagement():
    conn = get_db_connection()
    # Aggregate spend by household demographics
    demo_query = """
        SELECT hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN, SUM(tr.SPEND) AS TOTAL_SPEND
        FROM transactions tr
        INNER JOIN households hh ON tr.HSHD_NUM = hh.HSHD_NUM
        GROUP BY hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN
    """
    data_df = pd.read_sql_query(demo_query, conn)

    # Year-over-Year Spend
    yoy_query = """
        SELECT YEAR, SUM(SPEND) AS TOTAL_SPEND
        FROM transactions
        GROUP BY YEAR
        ORDER BY YEAR
    """
    yoy_df = pd.read_sql_query(yoy_query, conn)

    # Product Category Popularity
    cat_query = """
        SELECT pr.DEPARTMENT, SUM(tr.UNITS) AS TOTAL_UNITS
        FROM transactions tr
        INNER JOIN products pr ON tr.PRODUCT_NUM = pr.PRODUCT_NUM
        GROUP BY pr.DEPARTMENT
        ORDER BY TOTAL_UNITS DESC
    """
    cat_df = pd.read_sql_query(cat_query, conn)

    # Seasonal Spending Patterns
    seasonal_query = """
        SELECT WEEK_NUM, SUM(SPEND) AS TOTAL_SPEND
        FROM transactions
        GROUP BY WEEK_NUM
        ORDER BY WEEK_NUM
    """
    seasonal_df = pd.read_sql_query(seasonal_query, conn)

    # Brand & Organic Preferences
    brand_query = """
        SELECT pr.BRAND_TY, pr.NATURAL_ORGANIC_FLAG, SUM(tr.UNITS) AS TOTAL_UNITS
        FROM transactions tr
        INNER JOIN products pr ON tr.PRODUCT_NUM = pr.PRODUCT_NUM
        GROUP BY pr.BRAND_TY, pr.NATURAL_ORGANIC_FLAG
    """
    brand_df = pd.read_sql_query(brand_query, conn)

    conn.close()

    # --- Build Plotly charts ---
    # 1) Household Size vs Spend
    hh_df = data_df.groupby('HH_SIZE')['TOTAL_SPEND'].sum().reset_index()
    hh_size_fig = px.bar(hh_df, x='HH_SIZE', y='TOTAL_SPEND',
                         title='Household Size vs Total Spend')
    hh_size_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    hh_size_plot = hh_size_fig.to_html(full_html=False)

    # 2) Income Range vs Spend
    inc_df = data_df.groupby('INCOME_RANGE')['TOTAL_SPEND'].sum().reset_index()
    income_fig = px.bar(inc_df, x='INCOME_RANGE', y='TOTAL_SPEND',
                        title='Income Range vs Total Spend')
    income_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    income_plot = income_fig.to_html(full_html=False)

    # 3) Presence of Children vs Spend
    ch_df = data_df.groupby('CHILDREN')['TOTAL_SPEND'].sum().reset_index()
    children_fig = px.bar(ch_df, x='CHILDREN', y='TOTAL_SPEND',
                         title='Presence of Children vs Total Spend')
    children_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    children_plot = children_fig.to_html(full_html=False)

    # 4) Year-over-Year Spend
    yoy_fig = px.line(yoy_df, x='YEAR', y='TOTAL_SPEND',
                      title='Year-over-Year Household Spending', markers=True)
    yoy_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    yoy_spend_plot = yoy_fig.to_html(full_html=False)

    # 5) Category Popularity
    cat_fig = px.bar(cat_df, x='DEPARTMENT', y='TOTAL_UNITS',
                     title='Product Categories by Popularity')
    cat_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    category_popularity_plot = cat_fig.to_html(full_html=False)

    # 6) Seasonal Patterns
    seasonal_fig = px.line(seasonal_df, x='WEEK_NUM', y='TOTAL_SPEND',
                           title='Seasonal Spending Patterns', markers=True)
    seasonal_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    seasonal_plot = seasonal_fig.to_html(full_html=False)

    # 7) Brand & Organic Preferences
    brand_fig = px.bar(brand_df, x='BRAND_TY', y='TOTAL_UNITS',
                       color='NATURAL_ORGANIC_FLAG', barmode='group',
                       title='Brand & Organic Preferences')
    brand_fig.update_layout(margin=dict(l=20,r=20,t=40,b=20))
    brand_pref_plot = brand_fig.to_html(full_html=False)

    # Render the dashboard template
    return render_template(
        'demographicsandengagement.html',
        hh_size_plot=hh_size_plot,
        income_plot=income_plot,
        children_plot=children_plot,
        yoy_spend_plot=yoy_spend_plot,
        category_popularity_plot=category_popularity_plot,
        seasonal_plot=seasonal_plot,
        brand_pref_plot=brand_pref_plot
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
    # 1️⃣ Load data
    households, products, transactions = load_data()
    if households is None or products is None or transactions is None:
        return jsonify({'error': 'Failed to load data from database'}), 500

    # 2️⃣ Prepare features
    X, y, _ = prepare_features(households, products, transactions)

    # 3️⃣ Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4️⃣ Train
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # 5️⃣ Evaluate
    y_pred = gb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # 6️⃣ Save models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'gb_model_{timestamp}.pkl'
    joblib.dump(gb_model, model_filename)        # archived snapshot
    joblib.dump(gb_model, 'gb_model.pkl')       # canonical current model

    # 7️⃣ Persist metrics
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            IF NOT EXISTS (
              SELECT * FROM sysobjects WHERE name='model_metrics' AND xtype='U'
            )
            CREATE TABLE model_metrics (
              id INT IDENTITY(1,1) PRIMARY KEY,
              model_filename VARCHAR(255),
              mse FLOAT,
              timestamp DATETIME
            )
        """)
        cursor.execute(
            "INSERT INTO model_metrics (model_filename, mse, timestamp) VALUES (?, ?, GETDATE())",
            (model_filename, mse)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error storing metrics: {e}")

    # 8️⃣ Return result
    return jsonify({
        'mse': mse,
        'model_filename': model_filename
    })



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
    # 1) Look for any saved models
    candidates = glob.glob('gb_model_*.pkl')

    # 2) If none exist yet, trigger a train and then re-scan
    if not candidates:
        train_model()                         # trains & writes at least one gb_model_<ts>.pkl
        candidates = glob.glob('gb_model_*.pkl')

    # 3) If we still have none, error out
    if not candidates:
        return jsonify({ 'error': 'Model training failed; please try again.' }), 500

    # 4) Otherwise load the newest one
    latest_model = max(candidates, key=os.path.getctime)
    model = joblib.load(latest_model)

    # 5) Your existing prediction logic…
    households, products, transactions = load_data()
    X, _, merged = prepare_features(households, products, transactions)
    preds = model.predict(X)
    merged['predicted_spend'] = preds
    top = (
        merged
        .groupby(['DEPARTMENT','COMMODITY'])
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
