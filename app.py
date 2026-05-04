# ============================================================
# app.py — Dashboard Market Basket Analysis (Apriori)
# Dibuat dengan Streamlit untuk interaktivitas tinggi.
#
# Cara menjalankan:
#   streamlit run app.py
# ============================================================

# ── Import Library ──────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# KONFIGURASI HALAMAN (HARUS DIPANGGIL PERTAMA KALI)
# ─────────────────────────────────────────────────────────────
# st.set_page_config() wajib menjadi perintah Streamlit pertama.
# Fungsi:
#   - page_title : judul tab browser
#   - page_icon  : emoji/ikon favicon
#   - layout     : 'wide' memanfaatkan lebar penuh layar,
#                  cocok untuk dashboard dengan banyak kolom
#   - initial_sidebar_state : sidebar terbuka saat pertama load
# ============================================================
st.set_page_config(
    page_title="Market Basket Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS — Mempercantik tampilan tanpa framework eksternal
# ─────────────────────────────────────────────────────────────
# st.markdown() dengan unsafe_allow_html=True memungkinkan
# kita menyuntikkan HTML/CSS murni ke halaman Streamlit.
# Digunakan untuk elemen yang belum tersedia secara native.
# ============================================================
st.markdown("""
<style>
    /* Warna utama aplikasi */
    :root {
        --primary: #1E3A5F;
        --accent: #2196F3;
        --success: #4CAF50;
        --warning: #FF9800;
    }

    /* Header utama halaman */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2196F3 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Kartu metrik ringkasan */
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }

    /* Kotak penjelasan aturan terpilih */
    .rule-highlight {
        border-left: 5px solid #2196F3;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 1.05rem;
    }

    /* Styling sidebar */
    .sidebar-section {
        background: #F8F9FA;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border: 1px solid #E0E0E0;
    }
            
    /*table*/
    .table-container {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        max-height: 400px;   /* tinggi tabel */
        overflow-y: auto;    /* scroll vertikal */
        overflow-x: hidden;    /* scroll horizontal */
    }
            
    .custom-table {
        width: 100% !important;
        min-width: 100%;
    }

    .custom-table thead {
        color: white;
    }

    .custom-table th, .custom-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #2c2c2c;
    }

    /* WRAP TEXT */
    .custom-table td {
        white-space: normal !important;
        word-break: break-word;
        max-width: 300px;
    }

    /* Hover effect */
    .custom-table tbody tr:hover {
        background-color: rgba(33, 150, 243, 0.1);
        transition: 0.2s;
    }

    /* Highlight kolom support */
    .custom-table td:nth-child(2) {
        color: #4FC3F7;
        font-weight: 500;
    }

    /* Highlight kolom transaksi */
    .custom-table td:nth-child(3) {
        color: #81C784;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# FUNGSI LOAD DATA — Dengan @st.cache_data
# ─────────────────────────────────────────────────────────────
# @st.cache_data adalah DEKORATOR KACHING milik Streamlit.
#
# MENGAPA PENTING?
# Setiap kali user mengubah slider atau filter, Streamlit
# me-rerun seluruh script dari atas ke bawah.
# Tanpa cache, dataset akan dimuat ulang dari disk SETIAP KALI
# — bisa memakan 3-10 detik untuk file besar.
#
# Dengan @st.cache_data:
# - Load data hanya terjadi SEKALI pada eksekusi pertama.
# - Hasil disimpan di memori dan di-reuse pada rerun berikutnya.
# - Cache dianggap "basi" hanya jika argumen fungsi berubah.
# ============================================================
@st.cache_data(hash_funcs={type(None): lambda _: None})
def load_and_clean_data(uploaded_file) -> pd.DataFrame:
    """
    Memuat dan membersihkan dataset Online Retail dari uploaded file.
    Hasil di-cache agar tidak dimuat ulang setiap rerun.
    """
    # Deteksi format file berdasarkan nama file
    if uploaded_file is None:
        raise ValueError("File belum tersedia")

    # Baca file
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # 🔥 Cleaning versi sederhana (sesuai dataset kamu)
    df.dropna(subset=['InvoiceNo', 'Description', 'Country'], inplace=True)

    # Pastikan tipe data
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df['Description'] = df['Description'].astype(str)

    # Normalisasi teks
    df['Description'] = df['Description'].str.strip().str.upper()

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# FUNGSI BUAT BASKET MATRIX
# ─────────────────────────────────────────────────────────────
# Juga di-cache karena proses groupby + unstack + fillna
# bisa lambat untuk dataset besar (>500k baris).
#
# Argumen 'country' menjadi bagian dari cache key:
# - basket('UK')  → hasil cache A
# - basket('All') → hasil cache B
# Keduanya disimpan terpisah secara otomatis oleh Streamlit.
# ============================================================
@st.cache_data
def create_basket_matrix(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    Membuat One-Hot Encoded basket matrix dari data bersih.
    """
    # Filter negara
    if country != 'All Countries':
        df = df[df['Country'] == country]

    # Buat pivot dan encode ke boolean
    basket = (
        df.groupby(['InvoiceNo', 'Description'])
        .size()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index('InvoiceNo')
    )
    basket_encoded = basket.applymap(lambda x: True if x > 0 else False)

    # Hapus kolom yang semua nilainya False (produk tidak pernah dibeli)
    basket_encoded = basket_encoded.loc[:, basket_encoded.any()]

    return basket_encoded


# ============================================================
# FUNGSI JALANKAN APRIORI + GENERATE RULES
# ─────────────────────────────────────────────────────────────
# Di-cache dengan argumen support dan confidence sebagai key.
# Artinya: mengubah slider akan memicu komputasi ulang HANYA
# untuk kombinasi parameter yang belum pernah dihitung.
# ============================================================
@st.cache_data
def run_apriori(basket: pd.DataFrame, min_support: float, min_confidence: float):
    """
    Menjalankan algoritma Apriori dan menghasilkan association rules.

    Returns:
    --------
    frequent_itemsets : DataFrame itemset yang memenuhi min_support
    rules             : DataFrame aturan asosiasi
    """
    # ── Tahap 1: Cari frequent itemsets ──
    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=4
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(), pd.DataFrame()

    frequent_itemsets['itemset_length'] = frequent_itemsets['itemsets'].apply(len)

    # ── Tahap 2: Generate aturan asosiasi ──
    rules = association_rules(
        frequent_itemsets,
        metric='lift',
        min_threshold=1.0
    )

    if rules.empty:
        return frequent_itemsets, pd.DataFrame()

    # Filter berdasarkan confidence
    rules = rules[rules['confidence'] >= min_confidence]

    # Konversi frozenset ke string yang mudah dibaca
    rules['antecedents_str'] = rules['antecedents'].apply(
        lambda x: ' + '.join(sorted(list(x)))
    )
    rules['consequents_str'] = rules['consequents'].apply(
        lambda x: ' + '.join(sorted(list(x)))
    )

    # Format kolom numerik
    for col in ['support', 'confidence', 'lift', 'leverage', 'conviction']:
        if col in rules.columns:
            rules[col] = rules[col].round(4)

    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules


# ============================================================
# SIDEBAR — Panel Kontrol Utama
# ─────────────────────────────────────────────────────────────
# st.sidebar memisahkan kontrol dari konten utama.
#
# MENGAPA SIDEBAR?
# 1. Konvensi UX standar: kontrol di kiri, hasil di kanan.
# 2. Tidak memakan ruang konten utama yang berharga.
# 3. Streamlit otomatis menyembunyikannya di layar kecil (mobile).
# 4. Semua widget di sidebar bisa diakses dari mana saja
#    dalam script via st.sidebar.xxx
# ============================================================
with st.sidebar:
    # ── Upload File ──────────────────────────────────────────
    # st.file_uploader: Komponen upload file bawaan Streamlit.
    # Mengembalikan objek file-like yang bisa langsung dibaca pandas.
    # MENGAPA: Lebih aman dan portabel daripada hardcode path file.
    # User tidak perlu tahu di mana file disimpan di server.
    # ──────────────────────────────────────────────────────────
    st.markdown("### Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file Online Retail (CSV atau Excel)",
        type=['csv', 'xlsx']
    )

    # ── AUTO LOAD jika tidak upload ─────────────────────────────
    DEFAULT_PATH = "invoice_desc.xlsx"  # ← sesuaikan path kamu

    if uploaded_file is not None:
        df_clean = load_and_clean_data(uploaded_file)
    else:
        try:
            df_clean = load_and_clean_data(open(DEFAULT_PATH, 'rb'))
            st.success("Menggunakan dataset default")
        except FileNotFoundError:
            st.warning("Upload file atau pastikan dataset ada di path default.")
            st.stop()

    st.markdown("---")

    # ── Filter Negara ─────────────────────────────────────────
    # MENGAPA st.selectbox untuk negara?
    # - Pilihan negara bersifat kategorikal dan terbatas
    # - Selectbox lebih rapi daripada menampilkan 40+ tombol
    # - User bisa mengetik untuk mencari (built-in search)
    # ──────────────────────────────────────────────────────────
    st.markdown("### Filter Negara")

    # Load data terlebih dahulu untuk mengambil daftar negara
    countries = ['All Countries'] + sorted(df_clean['Country'].unique().tolist())

    selected_country = st.selectbox(
        "Pilih Negara",
        options=countries,
        index=0,
        help="Pilih 'All Countries' untuk analisis global, atau pilih negara tertentu."
    )

    # Tampilkan info jumlah transaksi untuk negara terpilih
    if selected_country != 'All Countries':
        n_transactions = df_clean[df_clean['Country'] == selected_country]['InvoiceNo'].nunique()
    else:
        n_transactions = df_clean['InvoiceNo'].nunique()
    st.caption(f"{n_transactions:,} invoice ditemukan")

    st.markdown("---")

    # ── Parameter Apriori ─────────────────────────────────────
    # MENGAPA st.slider untuk Support dan Confidence?
    # - Nilainya kontinu (desimal antara 0 dan 1)
    # - Slider memungkinkan user bereksperimen secara visual
    # - User bisa melihat langsung dampak perubahan nilai
    # - Lebih intuitif dari st.number_input untuk range 0-1
    #
    # format="%.3f" → tampilkan 3 desimal agar presisi terlihat
    # step=0.001    → granularitas perubahan per klik/drag
    # ──────────────────────────────────────────────────────────
    st.markdown("### Parameter Apriori")

    min_support = st.slider(
        "Minimum Support",
        min_value=0.005,
        max_value=0.1,
        value=0.02,
        step=0.005,
        format="%.3f",
        help=(
            "Seberapa sering item harus muncul agar dianggap 'frequent'.\n"
            "Nilai lebih kecil → lebih banyak aturan (lebih lambat).\n"
            "Nilai lebih besar → aturan lebih sedikit tapi lebih umum."
        )
    )

    min_confidence = st.slider(
        "Minimum Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        format="%.2f",
        help=(
            "Probabilitas minimum pembelian B jika A sudah dibeli.\n"
            "Nilai 0.3 = 30% kemungkinan pembelian bersamaan."
        )
    )

    # ── Tombol Jalankan ───────────────────────────────────────
    # MENGAPA st.button untuk memulai komputasi?
    # - Mencegah komputasi berat berjalan otomatis setiap
    #   kali slider digeser (lebih hemat resource)
    # - User memiliki kontrol penuh kapan analisis dijalankan
    # - Pola UX: atur parameter dulu, baru eksekusi
    # ──────────────────────────────────────────────────────────
    st.markdown("---")
    run_button = st.button(
        "Jalankan Analisis",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")


# ============================================================
# KONTEN UTAMA — Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>Market Basket Analysis Dashboard</h1>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PROSES ANALISIS (dijalankan saat tombol diklik)
# ─────────────────────────────────────────────────────────────
# Semua logika berat diletakkan di dalam blok if run_button.
# Ini mencegah Streamlit menjalankan Apriori setiap kali
# user menggeser slider atau mengubah filter lainnya.
# ============================================================
if run_button:
    # Simpan status "sudah dijalankan" ke session_state
    # st.session_state adalah dict yang persisten selama sesi
    # browser aktif — data tidak hilang meskipun script di-rerun
    st.session_state['analysis_run'] = True
    st.session_state['params'] = {
        'country': selected_country,
        'min_support': min_support,
        'min_confidence': min_confidence
    }


# Tampilkan hasil jika analisis sudah pernah dijalankan
if st.session_state.get('analysis_run', False):

    params = st.session_state['params']

    # ── Progress Indicator ────────────────────────────────────
    # st.spinner: Menampilkan animasi loading selama blok
    # kode di dalamnya sedang berjalan.
    # Penting untuk UX karena Apriori bisa memakan waktu 5-30 detik.
    # ──────────────────────────────────────────────────────────
    with st.spinner("⏳ Membangun basket matrix dan menjalankan Apriori..."):

        # Buat basket matrix
        basket = create_basket_matrix(df_clean, params['country'])

        if basket.empty:
            st.error("❌ Tidak ada data untuk negara yang dipilih.")
            st.stop()

        # Jalankan Apriori
        frequent_itemsets, rules = run_apriori(
            basket,
            params['min_support'],
            params['min_confidence']
        )

    # ── Tampilkan status hasil ───────────────────────────────
    if frequent_itemsets.empty:
        st.warning(
            "⚠️ Tidak ada frequent itemsets ditemukan. "
            "Coba turunkan nilai Minimum Support."
        )
        st.stop()

    if rules.empty:
        st.warning(
            "⚠️ Tidak ada aturan asosiasi yang memenuhi syarat. "
            "Coba turunkan nilai Minimum Confidence."
        )

    # ============================================================
    # KARTU METRIK RINGKASAN
    # ─────────────────────────────────────────────────────────────
    # st.columns: Membagi halaman menjadi kolom-kolom sejajar.
    # MENGAPA: Menampilkan KPI (Key Performance Indicator) secara
    # horizontal adalah standar UX dashboard modern.
    # Lebih efisien dalam penggunaan vertikal layar.
    # ============================================================
    st.markdown("### Ringkasan Hasil")
    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        (col1, f"{basket.shape[0]:,}", "Total Invoice"),
        (col2, f"{basket.shape[1]:,}", "Produk Unik"),
        (col3, f"{len(frequent_itemsets):,}", "Frequent Itemsets"),
        (col4, f"{len(rules):,}", "Aturan Asosiasi"),
        (col5, f"{rules['lift'].max():.2f}" if not rules.empty else "N/A", "Lift Tertinggi"),
    ]

    for col, value, label in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================
    # TAB NAVIGASI
    # ─────────────────────────────────────────────────────────────
    # st.tabs: Memisahkan konten berbeda ke dalam tab horizontal.
    # MENGAPA TAB daripada menumpuk semua konten?
    # 1. Mengurangi cognitive overload — user fokus satu hal
    # 2. Halaman tidak terlalu panjang/scroll
    # 3. Setiap tab bisa berisi banyak konten tanpa saling mengganggu
    # ============================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tabel Aturan",
        "Visualisasi",
        "Eksplorasi Aturan",
        "Frequent Itemsets"
    ])

    # ──────────────────────────────────────────────────────────
    # TAB 1: TABEL ATURAN ASOSIASI
    # ──────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Semua Aturan Asosiasi")

        if not rules.empty:
            # ── Filter tambahan dalam tabel ───────────────────
            # st.columns dalam konten utama untuk sub-filter
            f1, f2 = st.columns(2)
            with f1:
                # MENGAPA st.number_input di sini daripada slider?
                # Karena user mungkin ingin memasukkan nilai presisi
                # seperti "lift minimal 3.5" — lebih mudah diketik
                min_lift_filter = st.number_input(
                    "Filter: Lift minimum",
                    min_value=1.0,
                    value=1.0,
                    step=0.5,
                    format="%.1f"
                )
            with f2:
                # Pilih kolom yang ditampilkan
                show_cols = st.multiselect(
                    "Tampilkan kolom",
                    options=['antecedents_str', 'consequents_str', 'support',
                             'confidence', 'lift', 'leverage', 'conviction'],
                    default=['antecedents_str', 'consequents_str',
                             'support', 'confidence', 'lift']
                )

            # Terapkan filter lift
            filtered_rules = rules[rules['lift'] >= min_lift_filter]

            if show_cols:
                display_df = filtered_rules[show_cols].copy()

                # Rename kolom agar lebih deskriptif
                rename_map = {
                    'antecedents_str': 'Jika Membeli (Antecedent)',
                    'consequents_str': 'Maka Membeli (Consequent)',
                    'support': 'Support',
                    'confidence': 'Confidence',
                    'lift': 'Lift',
                    'leverage': 'Leverage',
                    'conviction': 'Conviction'
                }
                display_df.rename(columns=rename_map, inplace=True)

                # ── st.dataframe ──────────────────────────────
                # MENGAPA st.dataframe daripada st.table?
                # st.dataframe : Interaktif — bisa sort, scroll,
                #                resize kolom. Ideal untuk tabel besar.
                # st.table     : Statis — tampilan bersih tapi tidak
                #                bisa di-sort atau di-scroll.
                # Untuk association rules yang bisa ratusan baris,
                # st.dataframe jauh lebih usable.
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=420,
                    hide_index=True
                )

                st.caption(f"Menampilkan {len(filtered_rules)} dari {len(rules)} aturan")

                # ── Tombol Download ───────────────────────────
                # st.download_button: Mengeksport data ke CSV.
                # MENGAPA: User biasanya ingin membawa hasil analisis
                # ke Excel atau tools lain untuk presentasi.
                csv_export = filtered_rules.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇Download Hasil (CSV)",
                    data=csv_export,
                    file_name=f"association_rules_{params['country'].replace(' ', '_')}.csv",
                    mime='text/csv',
                    use_container_width=True
                )

    # ──────────────────────────────────────────────────────────
    # TAB 2: VISUALISASI
    # ──────────────────────────────────────────────────────────
    with tab2:
        if not rules.empty:
            v1, v2 = st.columns(2)

            # ── Plot 1: Scatter Support vs Confidence ─────────
            with v1:
                st.markdown("##### Support vs Confidence (ukuran = Lift)")

                # MENGAPA Plotly daripada Matplotlib?
                # Plotly menghasilkan grafik INTERAKTIF (hover, zoom, pan)
                # yang jauh lebih berguna dalam dashboard web.
                # Matplotlib hanya menghasilkan gambar statis.
                fig_scatter = px.scatter(
                    rules.head(200),   # Batasi 200 aturan untuk performa
                    x='support',
                    y='confidence',
                    size='lift',
                    color='lift',
                    color_continuous_scale='YlOrRd',
                    hover_data={
                        'antecedents_str': True,
                        'consequents_str': True,
                        'lift': ':.3f',
                        'support': ':.4f',
                        'confidence': ':.3f'
                    },
                    labels={
                        'support': 'Support',
                        'confidence': 'Confidence',
                        'lift': 'Lift'
                    },
                    title="Support vs Confidence"
                )
                fig_scatter.update_layout(height=380)
                st.plotly_chart(fig_scatter, use_container_width=True)

            # ── Plot 2: Top N Aturan berdasarkan Lift ─────────
            with v2:
                st.markdown("##### Top 15 Aturan Berdasarkan Lift")

                top_rules = rules.head(15).copy()
                top_rules['rule_label'] = (
                    top_rules['antecedents_str'].str[:25] +
                    " → " +
                    top_rules['consequents_str'].str[:20]
                )

                fig_bar = px.bar(
                    top_rules.sort_values('lift'),
                    x='lift',
                    y='rule_label',
                    orientation='h',
                    color='confidence',
                    color_continuous_scale='Blues',
                    labels={'lift': 'Lift', 'rule_label': 'Aturan', 'confidence': 'Confidence'},
                    title="Top 15 Aturan (Lift Tertinggi)"
                )
                fig_bar.update_layout(height=380, yaxis_title="")
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Plot 3: Distribusi Lift ────────────────────────
            st.markdown("##### Distribusi Nilai Lift")
            fig_hist = px.histogram(
                rules,
                x='lift',
                nbins=40,
                color_discrete_sequence=['#2196F3'],
                labels={'lift': 'Nilai Lift', 'count': 'Frekuensi'},
                title="Distribusi Lift — semakin ke kanan, semakin kuat hubungan antar produk"
            )
            fig_hist.add_vline(
                x=rules['lift'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Rata-rata: {rules['lift'].mean():.2f}"
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

    # ──────────────────────────────────────────────────────────
    # TAB 3: EKSPLORASI ATURAN (FITUR UTAMA)
    # ──────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Cari Aturan untuk Produk Tertentu")

        if not rules.empty:
            # Kumpulkan semua produk yang muncul dalam aturan
            all_products = sorted(set(
                rules['antecedents_str'].tolist() +
                rules['consequents_str'].tolist()
            ))

            # ── st.selectbox untuk produk ─────────────────────
            # MENGAPA TIDAK text_input?
            # selectbox mencegah user salah ketik nama produk
            # dan otomatis menampilkan pilihan yang tersedia.
            # Lebih user-friendly untuk non-technical user.
            selected_product = st.selectbox(
                "Pilih Produk (Antecedent)",
                options=all_products,
                help="Pilih produk untuk melihat produk apa yang sering dibeli bersamanya."
            )

            # Filter aturan yang mengandung produk terpilih sebagai antecedent
            product_rules = rules[
                rules['antecedents_str'].str.contains(
                    selected_product, case=False, na=False
                )
            ].head(10)

            if not product_rules.empty:
                st.markdown(f"##### Jika membeli **{selected_product}**, kemungkinan juga membeli:")
                st.markdown("")

                for _, row in product_rules.iterrows():
                    confidence_pct = row['confidence'] * 100
                    lift_val = row['lift']

                    # Tentukan warna berdasarkan kekuatan lift
                    if lift_val >= 5:
                        badge = "🔴 Sangat Kuat"
                    elif lift_val >= 3:
                        badge = "🟠 Kuat"
                    elif lift_val >= 2:
                        badge = "🟡 Sedang"
                    else:
                        badge = "🟢 Lemah"

                    st.markdown(f"""
                    <div class="rule-highlight">
                        <strong> -> {row['consequents_str']}</strong><br>
                        <small>
                            Confidence: <b>{confidence_pct:.1f}%</b> &nbsp;|&nbsp;
                            Lift: <b>{lift_val:.3f}</b> ({badge}) &nbsp;|&nbsp;
                            Support: <b>{row['support']:.4f}</b>
                        </small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Tidak ada aturan dengan '{selected_product}' sebagai antecedent.")

    # ──────────────────────────────────────────────────────────
    # TAB 4: FREQUENT ITEMSETS
    # ──────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Frequent Itemsets")

        # ── st.radio untuk filter panjang itemset ─────────────
        # MENGAPA st.radio daripada slider?
        # Panjang itemset adalah bilangan bulat kecil (1, 2, 3, 4).
        # Radio button lebih jelas secara visual daripada slider
        # untuk pilihan diskrit dengan jumlah sedikit.
        length_filter = st.radio(
            "Filter panjang itemset",
            options=['Semua', '1-item', '2-item', '3-item', '4-item'],
            horizontal=True
        )

        fi_display = frequent_itemsets.copy()
        fi_display['itemsets_str'] = fi_display['itemsets'].apply(
            lambda x: ' + '.join(sorted(list(x)))
        )

        if length_filter != 'Semua':
            length_map = {'1-item': 1, '2-item': 2, '3-item': 3, '4-item': 4}
            fi_display = fi_display[
                fi_display['itemset_length'] == length_map[length_filter]
            ]

        fi_display = fi_display.sort_values('support', ascending=False)

        col_fi1, col_fi2 = st.columns([2, 1])

        with col_fi1:
                fi_display['transaction_count'] = (
                    fi_display['support'] * basket.shape[0]
                ).astype(int)
                df_display = fi_display[['itemsets_str', 'support', 'transaction_count']].rename(columns={
                    'itemsets_str': 'Itemset',
                    'support': 'Support',
                    'transaction_count': 'Jumlah Transaksi'
                })

                st.markdown(
                    f"""
                    <div class="table-container">
                        {df_display.to_html(classes="custom-table", index=False, escape=False)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col_fi2:
            st.markdown("##### Jumlah per Ukuran Itemset")
            size_dist = frequent_itemsets['itemset_length'].value_counts().sort_index()
            fig_pie = px.pie(
                values=size_dist.values,
                names=[f"{i}-item" for i in size_dist.index],
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig_pie.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)

else:
    # ── Tampilan awal sebelum analisis dijalankan ────────────
    # st.info: Kotak pesan berwarna biru — informatif, non-urgent
    st.info("👈 Upload dataset dan klik **Jalankan Analisis** di sidebar untuk memulai.")

    # Tampilkan penjelasan singkat tentang algoritma
    with st.expander("ℹ️ Apa itu Market Basket Analysis?"):
        st.markdown("""
        **Market Basket Analysis (MBA)** adalah teknik data mining yang mengidentifikasi
        produk-produk yang sering dibeli bersamaan.

        **Metrik yang digunakan:**

        | Metrik | Rumus | Interpretasi |
        |--------|-------|--------------|
        | **Support** | P(A ∩ B) | Seberapa sering A dan B muncul bersama |
        | **Confidence** | P(B\|A) | Jika A dibeli, seberapa sering B juga dibeli |
        | **Lift** | Confidence / P(B) | Lift > 1: hubungan positif (bukan kebetulan) |

        **Cara baca aturan:** *"Jika pelanggan membeli **A**, maka kemungkinan besar juga membeli **B**"*
        """)