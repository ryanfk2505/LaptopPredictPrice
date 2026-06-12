# app.py - Laptop Recommendation System with Currency Converter
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Laptop Recommendation System",
    layout="wide"
)

# --- INJEKSI CSS KUSTOM ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    html, body, [data-testid="stSidebar"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #000000 !important;
    }
    .main-title {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    code {
        color: #FFFFFF !important;
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        padding: 0.1rem 0.4rem !important;
    }
    div[data-testid="stExpander"] {
        background-color: #0A0A0A !important;
        border: 1px solid #222222 !important;
        border-radius: 6px !important;
        margin-bottom: 0.8rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: #990000 !important;
    }
    div[data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    .filter-card {
        background: #0A0A0A;
        padding: 1.2rem;
        border-radius: 6px;
        border: 1px solid #222222;
    }
    .filter-card h4 {
        color: #990000 !important;
        margin-top: 0;
        border-bottom: 1px solid #222222;
        padding-bottom: 0.5rem;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .filter-card p {
        color: #CCCCCC !important;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }
    .price-badge {
        background-color: #1A0000;
        color: #FF3333;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid #440000;
    }
    button[data-testid="stBaseButton-primary"] {
        background-color: #990000 !important;
        border-color: #990000 !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #CC0000 !important;
        border-color: #CC0000 !important;
    }
    .footer-text {
        text-align: center;
        color: #444444;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    .error-text {
        color: #FF4444;
        background-color: #1A0000;
        padding: 0.5rem;
        border-radius: 4px;
        border-left: 3px solid #FF4444;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">Laptop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Temukan laptop terbaik sesuai budget dan spesifikasi kebutuhan Anda secara presisi.</p>', unsafe_allow_html=True)

# ============================================================
# FUNGSI KURS MATA UANG (DENGAN FALLBACK)
# ============================================================

def get_live_exchange_rates(base_currency='INR'):
    """Fetch live exchange rates from API dengan fallback"""
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        rates = data['rates']
        needed = ['IDR', 'USD', 'EUR', 'GBP', 'JPY', 'SGD', 'MYR']
        filtered = {k: rates.get(k, 1) for k in needed}
        filtered[base_currency] = 1
        
        return {
            'rates': filtered,
            'timestamp': datetime.now().isoformat(),
            'base': base_currency,
            'source': 'live'
        }
    except Exception as e:
        st.warning(f"Gagal mengambil kurs live, menggunakan data lokal. Error: {e}")
        return None

# Fallback rates
FALLBACK_RATES = {
    'INR': 1, 'IDR': 191.5, 'USD': 0.012, 'EUR': 0.011,
    'GBP': 0.0095, 'JPY': 1.85, 'SGD': 0.016, 'MYR': 0.056
}

# ============================================================
# LOAD DATA & MODEL
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('laptop_data.csv')
    
    # Cek dan hapus duplikat
    duplicates = df.duplicated(subset=['Model']).sum()
    if duplicates > 0:
        st.warning(f"Ditemukan {duplicates} duplikat data, akan dihapus")
        df = df.drop_duplicates(subset=['Model'])
    
    return df

@st.cache_resource
def load_models():
    df = pd.read_csv('laptop_data.csv')
    
    # Hapus duplikat
    df = df.drop_duplicates(subset=['Model'])
    
    numerical_cols = ['Price', 'RAM_GB', 'SSD_GB', 'Inches', 'Rating']
    categorical_cols = ['CPU_Detail', 'GPU_Detail', 'OS_Detail', 'Screen_Resolution_Type']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(df)

    knn_model = NearestNeighbors(
        n_neighbors=min(11, len(df)),
        metric='cosine'
    )
    knn_model.fit(X_processed)

    return knn_model, preprocessor

# Panggil
try:
    knn_model, preprocessor = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

def load_unique_values():
    with open('unique_values.json', 'r') as f:
        return json.load(f)

def convert_currency(amount_inr, from_currency='INR', to_currency='IDR', exchange_rates=None):
    if exchange_rates is None:
        return amount_inr
    if from_currency != 'INR':
        amount_inr = amount_inr / exchange_rates[from_currency]
    if to_currency != 'INR':
        return amount_inr * exchange_rates[to_currency]
    return amount_inr

def format_currency(amount, currency):
    symbols = {'INR': '₹', 'IDR': 'Rp', 'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥', 'SGD': 'S$', 'MYR': 'RM'}
    symbol = symbols.get(currency, '')
    if currency == 'IDR':
        return f"{symbol} {amount:,.0f}"
    return f"{symbol} {amount:,.2f}"

# Load semua file
try:
    df_clean = load_data()
    unique_vals = load_unique_values()
    
    # Coba ambil kurs live
    live_rates = get_live_exchange_rates('INR')
    if live_rates:
        exchange_rates = live_rates['rates']
        st.toast(f"✅ Kurs diperbarui: {live_rates['timestamp'][:10]}")
    else:
        exchange_rates = unique_vals.get('exchange_rates', FALLBACK_RATES)
        st.toast("⚠️ Menggunakan kurs lokal (offline mode)")
    
    st.toast("Sistem siap digunakan.")
except Exception as e:
    st.error(f"Error saat memuat sistem: {e}")
    st.stop()

# ============================================================
# FUNGSI VALIDASI INPUT
# ============================================================

def validate_budget(budget):
    if budget <= 0:
        st.sidebar.error("❌ Budget harus lebih dari 0!")
        return False
    if budget > 500_000_000:  # 500 juta IDR
        st.sidebar.warning("⚠️ Budget sangat besar, pastikan sesuai dengan kebutuhan")
    return True

def validate_rating(rating):
    if rating < 0 or rating > 100:
        st.sidebar.error("❌ Rating harus antara 0-100!")
        return False
    return True

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("### Filter Pencarian")

currency = st.sidebar.selectbox("Mata Uang", options=['IDR (Rupiah)', 'INR (Rupee)'], index=0)
currency_map = {'IDR (Rupiah)': 'IDR', 'INR (Rupee)': 'INR'}
selected_currency = currency_map[currency]

default_budget_inr = 50000
default_budget = convert_currency(default_budget_inr, 'INR', selected_currency, exchange_rates)
budget = st.sidebar.number_input(
    f"Budget ({format_currency(0, selected_currency)[0].strip()})",
    min_value=0.0,
    value=float(default_budget),
    step=1000000.0 if selected_currency == 'IDR' else 500.0
)

# Validasi budget
if not validate_budget(budget):
    st.stop()

ram_min = st.sidebar.selectbox("RAM ", options=[None, 4, 8, 16, 32], format_func=lambda x: "Semua Kapasitas" if x is None else f"{x} GB")

cpu_options = ['Semua'] + unique_vals.get('cpu_details', [])
cpu_detail = st.sidebar.selectbox("Prosesor (CPU)", options=cpu_options)
cpu_detail = None if cpu_detail == 'Semua' else cpu_detail

gpu_options = ['Semua'] + unique_vals.get('gpu_details', [])
gpu_detail = st.sidebar.selectbox("Kartu Grafis (GPU)", options=gpu_options)
gpu_detail = None if gpu_detail == 'Semua' else gpu_detail

use_screen_filter = st.sidebar.checkbox("Filter Ukuran Layar", value=False)
screen_size = None
if use_screen_filter:
    screen_size = st.sidebar.slider(
        "Ukuran Layar Minimal (inci)",
        min_value=10.0, max_value=18.0, value=13.0, step=0.1
    )

rating_min = st.sidebar.number_input(
    "Rating", 
    min_value=0, 
    max_value=100, 
    value=0, 
    step=5,
    help="Masukkan rating minimal (0-100), atau gunakan tombol +/-"
)

# Validasi rating
if not validate_rating(rating_min):
    st.stop()

n_recs = st.sidebar.slider("Jumlah Hasil Tampilan", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Metode Rekomendasi")
recommendation_method = st.sidebar.radio(
    "Pilih algoritma:",
    ["Filter Standar (Harga + Spesifikasi)", "🤖 KNN Machine Learning (Mirip dengan terbaik)"],
    help="KNN akan mencari laptop paling mirip dengan laptop rating tertinggi dalam budget Anda"
)

search_button = st.sidebar.button("Cari Laptop Terbaik", type="primary", use_container_width=True)

# Konversi budget ke INR
budget_inr = convert_currency(budget, selected_currency, 'INR', exchange_rates)

# ============================================================
# FUNGSI REKOMENDASI (DENGAN PERBAIKAN)
# ============================================================

def recommend_laptops(price_max_inr, ram_min=None, cpu_detail=None, gpu_detail=None, 
                       screen_size_min=None, rating_min=None, n=5):
    """Filter standar - dengan validasi parameter yang benar"""
    
    # Validasi parameter (fix bug untuk nilai 0)
    filtered = df_clean[df_clean['Price'] <= price_max_inr].copy()
    
    if ram_min is not None and ram_min > 0:
        filtered = filtered[filtered['RAM_GB'] >= ram_min]
    if cpu_detail is not None and cpu_detail != '':
        filtered = filtered[filtered['CPU_Detail'].str.contains(cpu_detail, case=False, na=False)]
    if gpu_detail is not None and gpu_detail != '':
        filtered = filtered[filtered['GPU_Detail'].str.contains(gpu_detail, case=False, na=False)]
    if screen_size_min is not None and screen_size_min > 0:
        filtered = filtered[filtered['Inches'] >= screen_size_min]
    if rating_min is not None and rating_min > 0:
        filtered = filtered[filtered['Rating'] >= rating_min]
    
    if len(filtered) == 0:
        return pd.DataFrame()
    
    return filtered.sort_values('Price').head(n).reset_index(drop=True)

def recommend_knn(budget_inr, ram_min, rating_min, n_recommendations=5,
                  cpu_detail=None, gpu_detail=None, screen_size_min=None):
    
    numerical_cols = ['Price', 'RAM_GB', 'SSD_GB', 'Inches', 'Rating']
    categorical_cols = ['CPU_Detail', 'GPU_Detail', 'OS_Detail', 'Screen_Resolution_Type']

    # Filter calon laptop (dengan validasi parameter yang benar)
    candidates = df_clean[df_clean['Price'] <= budget_inr].copy()
    
    if ram_min is not None and ram_min > 0:
        candidates = candidates[candidates['RAM_GB'] >= ram_min]
    if rating_min is not None and rating_min > 0:
        candidates = candidates[candidates['Rating'] >= rating_min]
    if cpu_detail is not None and cpu_detail != '':
        candidates = candidates[candidates['CPU_Detail'].str.contains(cpu_detail, case=False, na=False)]
    if gpu_detail is not None and gpu_detail != '':
        candidates = candidates[candidates['GPU_Detail'].str.contains(gpu_detail, case=False, na=False)]
    if screen_size_min is not None and screen_size_min > 0:
        candidates = candidates[candidates['Inches'] >= screen_size_min]

    if len(candidates) == 0:
        return pd.DataFrame(), None

    # Pilih laptop referensi (rating tertinggi)
    best_idx = candidates['Rating'].idxmax()
    reference = df_clean.loc[best_idx]

    # Siapkan query point
    reference_df = pd.DataFrame(
        [reference[categorical_cols + numerical_cols].values],
        columns=categorical_cols + numerical_cols
    )
    query_processed = preprocessor.transform(reference_df)

    # Cari laptop mirip
    distances, indices = knn_model.kneighbors(
        query_processed,
        n_neighbors=min(n_recommendations + 5, len(df_clean))
    )

    similar_laptops = []
    for i, idx in enumerate(indices[0]):
        # Skip jika ini adalah laptop referensi itu sendiri (gunakan index, bukan nama)
        if idx == best_idx:
            continue
        
        laptop = df_clean.iloc[idx]
        laptop_dict = laptop.to_dict()
        laptop_dict['Similarity'] = 1 - distances[0][i]
        similar_laptops.append(laptop_dict)
        
        if len(similar_laptops) >= n_recommendations:
            break

    # Jika kurang, hitung similarity secara eksplisit untuk fallback
    if len(similar_laptops) < n_recommendations:
        remaining = [i for i in candidates.index if i not in [l['index'] for l in similar_laptops] if i != best_idx]
        for idx in remaining[:n_recommendations - len(similar_laptops)]:
            laptop = df_clean.loc[idx]
            laptop_dict = laptop.to_dict()
            
            # Hitung similarity secara eksplisit
            laptop_df = pd.DataFrame(
                [laptop[categorical_cols + numerical_cols].values],
                columns=categorical_cols + numerical_cols
            )
            laptop_processed = preprocessor.transform(laptop_df)
            sim = cosine_similarity(query_processed, laptop_processed)[0][0]
            
            laptop_dict['Similarity'] = sim
            similar_laptops.append(laptop_dict)

    return pd.DataFrame(similar_laptops), reference

# ============================================================
# MAIN CONTENT
# ============================================================

col1, col2 = st.columns([1, 2.5], gap="large")

with col1:
    st.markdown(f"""
    <div class="filter-card">
        <h4>Parameter Aktif</h4>
        <p><b>Budget:</b> {format_currency(budget, selected_currency)}</p>
        <p><b>RAM Minimal:</b> {f"{ram_min} GB" if ram_min else "Semua"}</p>
        <p><b>Spesifikasi CPU:</b> {cpu_detail if cpu_detail else "Semua"}</p>
        <p><b>Spesifikasi GPU:</b> {gpu_detail if gpu_detail else "Semua"}</p>
        <p><b>Ukuran Layar:</b> {f"{screen_size} Inci" if screen_size else "Semua Ukuran"}</p>
        <p><b>Rating Produk:</b> {rating_min} / 100</p>
        <p><b>Metode:</b> {recommendation_method}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if search_button:
        with st.spinner("Menganalisis database laptop..."):
            
            if recommendation_method == "Filter Standar (Harga + Spesifikasi)":
                results = recommend_laptops(budget_inr, ram_min, cpu_detail, gpu_detail, screen_size, rating_min, n_recs)
                
                if len(results) > 0:
                    st.markdown(f"### Hasil Pencarian ({len(results)} Laptop Ditemukan)")
                    st.caption("Diurutkan berdasarkan harga termurah")
                    
                    for idx, row in results.iterrows():
                        price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                        expander_title = f"{row['Model'][:55]}  {format_currency(price_conv, selected_currency)}"
                        
                        with st.expander(expander_title):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**Harga:** {format_currency(price_conv, selected_currency)}")
                                st.markdown(f"**RAM:** `{row['RAM_GB']:.0f} GB`")
                                st.markdown(f"**SSD:** `{row['SSD_GB']:.0f} GB`")
                            with c2:
                                st.markdown(f"**Layar:** `{row['Inches']:.1f}\"`")
                                st.markdown(f"**CPU:** `{row['CPU_Detail'][:50]}`")
                                st.markdown(f"**Rating:** {row['Rating']:.1f} / 100")
                else:
                    st.error("Tidak ada laptop yang sesuai dengan kriteria filter Anda.")
            
            else:  # KNN Method
                results_knn, reference = recommend_knn(
                    budget_inr, 
                    ram_min, 
                    rating_min, 
                    n_recs,
                    cpu_detail, 
                    gpu_detail, 
                    screen_size
                )
                
                if len(results_knn) > 0 and reference is not None:
                    st.markdown("### Rekomendasi KNN (Machine Learning)")
                    st.caption("Berdasarkan laptop dengan rating tertinggi dalam budget Anda")
                    
                    ref_price = convert_currency(reference['Price'], 'INR', selected_currency, exchange_rates)
                    
                    with st.expander("🏆 LAPTOP REFERENSI (Rating Tertinggi)", expanded=True):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**Model:** {reference['Model'][:70]}")
                            st.markdown(f"**Rating:** {reference['Rating']:.1f} / 100")
                            st.markdown(f"**Harga:** {format_currency(ref_price, selected_currency)}")
                        with c2:
                            st.markdown(f"**RAM:** {reference['RAM_GB']:.0f} GB")
                            st.markdown(f"**SSD:** {reference['SSD_GB']:.0f} GB")
                            st.markdown(f"**Layar:** {reference['Inches']:.1f}\"")
                    
                    st.markdown(f"#### 🔍 Laptop Paling Mirip (Top {len(results_knn)})")
                    
                    for idx, row in results_knn.iterrows():
                        price_conv = convert_currency(row['Price'], 'INR', selected_currency, exchange_rates)
                        similarity_pct = row['Similarity'] * 100
                        
                        expander_title = f"{row['Model'][:50]} {format_currency(price_conv, selected_currency)}  🎯 {similarity_pct:.1f}% mirip"
                        
                        with st.expander(expander_title):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown(f"**Harga:** {format_currency(price_conv, selected_currency)}")
                                st.markdown(f"**RAM:** {row['RAM_GB']:.0f} GB")
                                st.markdown(f"**SSD:** {row['SSD_GB']:.0f} GB")
                                st.markdown(f"**Tingkat Kemiripan:** {similarity_pct:.1f}%")
                            with c2:
                                st.markdown(f"**Layar:** {row['Inches']:.1f}\"")
                                st.markdown(f"**Rating:** {row['Rating']:.1f} / 100")
                                st.markdown(f"**CPU:** {row['CPU_Detail'][:45]}")
                else:
                    st.warning("Tidak ada cukup data untuk rekomendasi KNN. Coba tingkatkan budget atau kurangi filter.")
    else:
        st.info("👈 Atur preferensi Anda di panel sebelah kiri, kemudian klik tombol **'Cari Laptop Terbaik'**.")

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #222222; margin-top: 5rem;">
    <p class="footer-text">Powered by Streamlit Engine & Scikit-Learn KNN Algorithm • 2026 Laptop Recommendation System</p>
""", unsafe_allow_html=True)
