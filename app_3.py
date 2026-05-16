import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import io
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import shutil
import logging
from datetime import datetime
from utils import detect_anomalies, root_cause_analysis, compare_best_run, generate_pdf_report
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s — %(levelname)s — %(message)s',
)

# ── CSV persistence paths ─────────────────────────────────────────────────────
CSV_PATH       = 'microstructure_history.csv'
IMAGES_DIR     = 'saved_images'
CSV_COLUMNS    = [
    'timestamp', 'part_number', 'operateur',
    'spindle_speed', 'friction_force', 'upset_force',
    'friction_burnoff', 'total_burnoff',
    'martensite_pct', 'ferrite_pct', 'perlite_pct',
    'bainite_pct', 'autres_pct',
    'statut', 'image_filename',
]


def init_csv():
    """Create CSV file with headers if it doesn't exist."""
    try:
        os.makedirs(IMAGES_DIR, exist_ok=True)
        if not os.path.isfile(CSV_PATH):
            df = pd.DataFrame(columns=CSV_COLUMNS)
            df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    except Exception as e:
        logging.error(f"init_csv: {e}")
        st.error(f"❌ Impossible d'initialiser le fichier CSV : {e}")


def save_analysis(data_dict, image_pil, original_filename):
    """Append one analysis row to CSV and backup the image."""
    try:
        # Backup image
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = os.path.splitext(original_filename)[1] or '.png'
        saved_name = f"{data_dict['part_number']}_{ts_str}{ext}"
        saved_path = os.path.join(IMAGES_DIR, saved_name)
        image_pil.save(saved_path)
        data_dict['image_filename'] = saved_name

        # Append to CSV
        row_df = pd.DataFrame([data_dict], columns=CSV_COLUMNS)
        header_needed = not os.path.isfile(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
        row_df.to_csv(CSV_PATH, mode='a', index=False,
                      header=header_needed, encoding='utf-8-sig')
        return True
    except Exception as e:
        logging.error(f"save_analysis: {e}")
        st.error(f"❌ Erreur lors de l'enregistrement : {e}")
        return False


def load_history():
    """Load full CSV history into a DataFrame."""
    try:
        if not os.path.isfile(CSV_PATH):
            return pd.DataFrame(columns=CSV_COLUMNS)
        df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        if df.empty:
            return pd.DataFrame(columns=CSV_COLUMNS)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"load_history: {e}")
        st.error(f"❌ Erreur de lecture de l'historique : {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)


def export_csv_button():
    """Display a download button for the full CSV."""
    df = load_history()
    if df.empty:
        st.info("Aucune donnée à exporter.")
        return
    csv_bytes = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    today = datetime.now().strftime('%Y%m%d')
    st.download_button(
        label="📥 Exporter l'historique complet",
        data=csv_bytes,
        file_name=f"nexteer_fw_history_{today}.csv",
        mime='text/csv',
        use_container_width=True,
    )


# ── Machine Learning ──────────────────────────────────────────────────────────
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

def recommend_params(part_number, df_history):
    if df_history.empty: return None, None
    df_ok = df_history[(df_history['part_number'] == part_number) & (df_history['statut'] == 'OK')]
    if df_ok.empty: return None, None
    
    df_top = df_ok.nsmallest(3, 'martensite_pct')
    
    weights = 1 / (df_top['martensite_pct'] + 1e-5)
    weights /= weights.sum()
    
    params = {}
    for col in ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff']:
        params[col] = np.average(df_top[col], weights=weights)
    
    best_martensite = df_top.iloc[0]['martensite_pct']
    return params, best_martensite

def train_model(df_history):
    if len(df_history) < 10: return None
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        features = ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff']
        df_clean = df_history.dropna(subset=features + ['martensite_pct'])
        if len(df_clean) < 10: return None
        
        X = df_clean[features].values
        y = df_clean['martensite_pct'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        score = model.score(X_scaled, y)
        
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        return score
    except Exception as e:
        logging.error(f"train_model: {e}")
        if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
        if os.path.exists(SCALER_PATH): os.remove(SCALER_PATH)
        return None

def predict_martensite(params_dict):
    try:
        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(SCALER_PATH):
            return None, None
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        features = ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff']
        X = np.array([[params_dict.get(f, 0.0) for f in features]])
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        preds = [tree.predict(X_scaled)[0] for tree in model.estimators_]
        std = np.std(preds)
        
        return prediction, std
    except Exception as e:
        logging.error(f"predict_martensite: {e}")
        return None, None


# Initialize CSV on app start
init_csv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexteer — Analyse Microstructure",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,300;0,400;0,500;0,700;1,700;1,900&family=Source+Code+Pro:wght@400;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    color: #1A1A1A;
}

/* Background */
.stApp {
    background: #FFFFFF;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #F5F5F5 !important;
    border-right: 2px solid #E31837 !important;
}
section[data-testid="stSidebar"] * { color: #1A1A1A !important; }
section[data-testid="stSidebar"] .stMarkdown hr {
    border-color: rgba(227,24,55,0.15) !important;
}

/* ── Header banner (Nexteer official style) ── */
.nxt-header {
    background: #FFFFFF;
    border-bottom: 3px solid #E31837;
    border-radius: 0;
    padding: 18px 28px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.nxt-logo {
    font-family: 'Roboto', sans-serif;
    font-size: 38px;
    font-weight: 900;
    font-style: italic;
    color: #E31837;
    letter-spacing: -1px;
    line-height: 1;
}
.nxt-sub {
    font-family: 'Roboto', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 6px;
    color: #1A1A1A;
    text-transform: uppercase;
    margin-top: 2px;
}
.nxt-divider {
    width: 2px; height: 48px;
    background: #E31837;
    margin: 0 12px;
    border-radius: 1px;
}
.nxt-module {
    font-family: 'Roboto', sans-serif;
    font-size: 14px; font-weight: 700;
    letter-spacing: 1.5px; color: #1A1A1A;
    text-transform: uppercase;
}
.nxt-module-sub {
    font-size: 12px; color: #666666; margin-top: 4px;
    font-weight: 400;
}

/* ── Metric cards with left red accent ── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E0E0E0;
    border-left: 4px solid #E31837;
    border-radius: 4px;
    padding: 18px 20px;
    text-align: center;
    position: relative;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 12px rgba(227,24,55,0.10);
}
.metric-name {
    font-family: 'Source Code Pro', monospace;
    font-size: 10px; letter-spacing: 2px;
    color: #666666; text-transform: uppercase; margin-bottom: 8px;
}
.metric-val {
    font-family: 'Roboto', sans-serif;
    font-size: 28px; font-weight: 700; color: #1A1A1A;
}
.metric-unit { font-size: 13px; color: #666666; margin-left: 3px; font-weight: 400; }

/* ── Status banners ── */
.status-ok {
    background: #F0FFF4;
    border: 2px solid #00C851;
    border-left: 5px solid #00C851;
    border-radius: 4px;
    padding: 18px 24px;
    font-family: 'Roboto', sans-serif;
    font-size: 20px; font-weight: 700;
    letter-spacing: 2px; color: #00C851;
    text-align: center; text-transform: uppercase;
}
.status-nok {
    background: #FFF5F5;
    border: 2px solid #E31837;
    border-left: 5px solid #E31837;
    border-radius: 4px;
    padding: 18px 24px;
    font-family: 'Roboto', sans-serif;
    font-size: 20px; font-weight: 700;
    letter-spacing: 2px; color: #E31837;
    text-align: center; text-transform: uppercase;
}
.status-sub {
    font-size: 13px; margin-top: 6px;
    font-weight: 400; letter-spacing: 1px;
    color: #1A1A1A;
}

/* ── Section titles with left red bar ── */
.section-title {
    font-family: 'Roboto', sans-serif;
    font-size: 13px; font-weight: 700;
    letter-spacing: 3px; color: #E31837;
    text-transform: uppercase;
    border-bottom: none;
    border-left: 4px solid #E31837;
    padding: 6px 0 6px 12px;
    margin-bottom: 16px;
    background: linear-gradient(90deg, rgba(227,24,55,0.04), transparent);
}

/* ── Slider label ── */
.slider-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 11px; color: #666666; letter-spacing: 1px;
}

/* ── Streamlit buttons — Nexteer red ── */
.stButton > button {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Roboto', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    transition: background-color 0.2s !important;
}
.stButton > button:hover {
    background-color: #CC0000 !important;
    color: #FFFFFF !important;
}
.stButton > button:disabled {
    background-color: #CCCCCC !important;
    color: #FFFFFF !important;
}

/* ── Streamlit slider — red track ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #E31837 !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
    background: rgba(227,24,55,0.2) !important;
}

/* ── Streamlit file uploader border ── */
.stFileUploader section {
    border-color: #E31837 !important;
}

/* ── Footer — solid Nexteer red like the real site ── */
.nxt-footer {
    margin-top: 40px;
    background: #E31837;
    padding: 18px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Roboto', sans-serif;
    font-size: 11px;
    color: #FFFFFF;
    letter-spacing: 1px;
    border-radius: 0 0 4px 4px;
}
.nxt-footer a { color: #FFFFFF; text-decoration: underline; }

/* ── Force all form inputs to white background ── */
input, textarea, select,
[data-baseweb="input"] input,
[data-baseweb="base-input"] input,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 3px !important;
    -webkit-text-fill-color: #1A1A1A !important;
}
/* Focus state — red border */
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #E31837 !important;
    box-shadow: 0 0 0 1px #E31837 !important;
}
/* Number input wrapper & buttons */
.stNumberInput [data-baseweb="input"] {
    background-color: #FFFFFF !important;
}
.stNumberInput button {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
    border-color: #CCCCCC !important;
}
.stNumberInput button:hover {
    background-color: #E31837 !important;
    color: #FFFFFF !important;
}
/* Placeholder text */
.stTextInput input::placeholder,
.stNumberInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #999999 !important;
    -webkit-text-fill-color: #999999 !important;
}
/* File uploader area */
.stFileUploader section {
    background-color: #FFFFFF !important;
    border: 1px dashed #E31837 !important;
}
.stFileUploader section > * { color: #1A1A1A !important; }
/* Expander */
.streamlit-expanderHeader {
    background-color: #F5F5F5 !important;
    color: #1A1A1A !important;
    border-radius: 3px !important;
}
/* Warning & info boxes text */
.stAlert > div { color: #1A1A1A !important; }
/* Slider labels */
.stSlider label, .stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    color: #1A1A1A !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nxt-header">
    <div>
        <div class="nxt-logo">nexteer</div>
        <div class="nxt-sub">Automotive</div>
    </div>
    <div class="nxt-divider"></div>
    <div>
        <div class="nxt-module">Contrôle Qualité · Analyse Microstructurale</div>
        <div class="nxt-module-sub">Détection de Phase Martensitique — Seuillage Adaptatif</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Paramètres</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Image de microstructure",
        type=["jpg", "jpeg", "png", "tiff", "bmp"],
        help="Formats acceptés : JPEG, PNG, TIFF, BMP"
    )

    st.divider()

    # ── Section 1 — Identification pièce ──────────────────────────────────
    st.markdown('<div class="section-title">🏭 Identification Pièce</div>', unsafe_allow_html=True)

    part_number = st.text_input(
        "Part Number *",
        placeholder="ex: NXT-2024-001",
        help="Référence unique de la pièce (obligatoire)",
        key="part_number_input"
    )
    operateur = st.text_input(
        "Nom de l'opérateur",
        placeholder="ex: Ahmed B.",
        help="Opérateur réalisant l'analyse"
    )

    history_df = load_history()

    if part_number.strip():
        rec_params, best_mart = recommend_params(part_number.strip(), history_df)
        if rec_params:
            st.info(f"""
📌 **Paramètres recommandés pour {part_number.strip()}** :
• Spindle Speed : {rec_params['spindle_speed']:.0f} RPM
• Friction Force : {rec_params['friction_force']:.1f} kN
• Upset Force : {rec_params['upset_force']:.1f} kN
• Friction Burnoff : {rec_params['friction_burnoff']:.1f} mm
• Total Burn-off : {rec_params['total_burnoff']:.1f} mm

→ Basé sur le meilleur run : **{best_mart:.2f}% de martensite**
""")
        else:
            st.info("ℹ Aucun historique OK pour ce PN")

    st.divider()

    # ── Section 2 — Paramètres machine Friction Welding ───────────────────
    st.markdown('<div class="section-title">⚙ Paramètres Friction Welding</div>', unsafe_allow_html=True)

    spindle_speed = st.number_input(
        "Spindle Speed (RPM)",
        min_value=0, max_value=5000, step=10,
        help="Vitesse de rotation de la broche",
        key="spindle_speed_input"
    )
    friction_force = st.number_input(
        "Friction Force (kN)",
        min_value=0.0, max_value=200.0, step=0.5,
        help="Force de friction appliquée",
        key="friction_force_input"
    )
    upset_force = st.number_input(
        "Upset Force (kN)",
        min_value=0.0, max_value=200.0, step=0.5,
        help="Force de refoulement",
        key="upset_force_input"
    )
    friction_burnoff = st.number_input(
        "Friction Burn-off (mm)",
        min_value=0.0, max_value=50.0, step=0.1,
        help="Raccourcissement pendant la phase de friction",
        key="friction_burnoff_input"
    )
    total_burnoff = st.number_input(
        "Total Burn-off (mm)",
        min_value=0.0, max_value=50.0, step=0.1,
        help="Raccourcissement total (friction + refoulement)",
        key="total_burnoff_input"
    )

    # ── Validation warnings & Predictions ─────────────────────────────────
    machine_params = {
        "spindle_speed": spindle_speed,
        "friction_force": friction_force,
        "upset_force": upset_force,
        "friction_burnoff": friction_burnoff,
        "total_burnoff": total_burnoff,
    }
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Prédiction IA</div>', unsafe_allow_html=True)
    
    pred_pct, pred_std = predict_martensite(machine_params)
    if pred_pct is not None:
        delta_str = "🟢 Conforme" if pred_pct <= 30 else "🔴 Risque NOK"
        st.metric("Martensite prédite", f"{pred_pct:.2f}% ± {pred_std:.2f}%", delta=delta_str, delta_color="normal" if pred_pct <= 30 else "inverse")
    else:
        req_remaining = max(0, 10 - len(history_df))
        st.caption(f"⏳ {req_remaining} analyses restantes pour activer la prédiction IA")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Validation warnings ───────────────────────────────────────────────
    machine_params = {
        "Spindle Speed": spindle_speed,
        "Friction Force": friction_force,
        "Upset Force": upset_force,
        "Friction Burn-off": friction_burnoff,
        "Total Burn-off": total_burnoff,
    }
    zero_params = [name for name, val in machine_params.items() if val == 0]
    if zero_params:
        st.warning(f"⚠ Valeur à 0 : {', '.join(zero_params)}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Vérification Paramètres (IA)</div>', unsafe_allow_html=True)
    detect_anomalies(machine_params, history_df, part_number.strip())


    if not part_number.strip():
        st.info("ℹ Le **Part Number** est obligatoire pour lancer l'analyse.")

    st.divider()

    # ── Section 3 — Seuil d'analyse ───────────────────────────────────────
    st.markdown('<div class="section-title">🔬 Seuil d\'Analyse</div>', unsafe_allow_html=True)

    seuil = st.slider(
        "Seuil de luminosité (blanc)",
        min_value=0, max_value=255, value=120, step=1,
        help="Pixels > seuil → identifiés comme martensite (zones claires)"
    )
    st.markdown(f'<div class="slider-label">Valeur seuil : <b style="color:#E31837">{seuil}</b> / 255</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Source Code Pro',monospace;font-size:10px;color:#1A1A1A;line-height:2;">
    SEUIL REJET &nbsp;&nbsp; <b style="color:#E31837">&gt; 30%</b><br>
    ALGORITHME &nbsp;<b style="color:#E31837">SEUILLAGE NIVEAUX GRIS</b><br>
    NORME &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color:#E31837">ASTM E3-11</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Récapitulatif avant analyse ───────────────────────────────────────
    with st.expander("📋 Récapitulatif", expanded=False):
        st.markdown(f"""
| Champ | Valeur |
|-------|--------|
| **Part Number** | `{part_number if part_number.strip() else '—'}` |
| **Opérateur** | `{operateur if operateur.strip() else '—'}` |
| **Spindle Speed** | `{spindle_speed} RPM` |
| **Friction Force** | `{friction_force} kN` |
| **Upset Force** | `{upset_force} kN` |
| **Friction Burn-off** | `{friction_burnoff} mm` |
| **Total Burn-off** | `{total_burnoff} mm` |
| **Seuil** | `{seuil} / 255` |
| **Image** | `{'✔ Chargée' if uploaded else '✘ Aucune'}` |
        """)

    st.divider()

    # ── Bouton — bloqué si Part Number vide ou pas d'image ────────────────
    can_analyse = (uploaded is not None) and bool(part_number.strip())
    analyser = st.button("▶ Lancer l'Analyse", use_container_width=True,
                         disabled=(not can_analyse))


# ── Tabs structure ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Analyse", "📊 Dashboard", "📋 Historique", "🤖 Recommandations"])

with tab1:
    mode = st.radio("Mode d'analyse", ["Pièce unique", "Batch (plusieurs images)"], horizontal=True)
    
    if mode == "Batch (plusieurs images)":
        st.info("Mode Batch sélectionné. Chargez plusieurs images dans le menu latéral.")
        uploaded_files = st.file_uploader("Uploader plusieurs images", type=['jpg','png','tif','jpeg','bmp'], accept_multiple_files=True, key="batch_uploader")
        
        if uploaded_files and st.button("▶ Lancer Batch"):
            progress_bar = st.progress(0)
            batch_results = []
            
            for i, f in enumerate(uploaded_files):
                img_pil = Image.open(f)
                img_rgb = np.array(img_pil.convert('RGB'))
                grille = np.array(img_pil.convert('L'))
                img_hsv = cv2.cvtColor(cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), (5,5), 0), cv2.COLOR_BGR2HSV)
                
                # compute martensite
                mask_m = cv2.inRange(img_hsv, np.array([0, 0, seuil]), np.array([180, 60, 255]))
                mask_m = cv2.morphologyEx(cv2.morphologyEx(mask_m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                
                mart_pct = (cv2.countNonZero(mask_m) / grille.size) * 100
                
                # compute ferrite
                mask_f = cv2.inRange(img_hsv, np.array([0, 0, int(seuil*0.75)]), np.array([180, 40, seuil-1]))
                mask_f = cv2.morphologyEx(cv2.morphologyEx(mask_f, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                ferrite_pct = (cv2.countNonZero(mask_f) / grille.size) * 100
                
                statut = 'OK' if mart_pct <= 30 else 'NOK'
                batch_results.append({'Image': f.name, 'Martensite%': round(mart_pct, 2), 'Ferrite%': round(ferrite_pct, 2), 'Statut': statut})
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            df_batch = pd.DataFrame(batch_results)
            st.dataframe(df_batch, use_container_width=True)
            csv_batch = df_batch.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("📥 Exporter CSV Batch", data=csv_batch, file_name="batch_results.csv", mime="text/csv")
            
        st.stop()  # Stop execution for normal mode if batch is selected
        
    if uploaded is None:
        # Waiting state
        col_w = st.columns([1, 2, 1])[1]
        with col_w:
            st.markdown("""
            <div style="text-align:center;padding:80px 0;color:#666666;">
                <div style="font-size:64px;opacity:0.2;margin-bottom:20px;color:#E31837;">⬡</div>
                <div style="font-family:'Rajdhani',sans-serif;font-size:18px;letter-spacing:3px;text-transform:uppercase;color:#1A1A1A;">
                    En attente d'une image
                </div>
                <div style="font-size:13px;margin-top:10px;opacity:0.7;color:#1A1A1A;">
                    Chargez une image de microstructure dans le panneau latéral pour démarrer.
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        with st.spinner("🔬 Analyse microstructurale en cours..."):

            # ── Load & prepare image ──────────────────────────────────────────
            image_pil = Image.open(uploaded)
            img_rgb   = np.array(image_pil.convert('RGB'))
            grille    = np.array(image_pil.convert('L'))

            # Convert to BGR for OpenCV, then to HSV
            img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_blur  = cv2.GaussianBlur(img_bgr, (5, 5), 0)
            img_hsv   = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

            total_pixels = grille.size
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # ── Phase definitions (name, HSV_low, HSV_high, overlay_BGR, critical%) ──
            phases = [
                ("Martensite",           np.array([0, 0, seuil]),  np.array([180, 60, 255]),  (23, 55, 255),    30.0),
                ("Ferrite",              np.array([0, 0, int(seuil*0.75)]), np.array([180, 40, seuil-1]), (245, 165, 66),   None),
                ("Perlite",              np.array([0, 0, 0]),      np.array([180, 50, int(seuil*0.75)-1]),  (99, 110, 141),   None),
                ("Bainite",              np.array([0, 20, 80]),    np.array([180, 80, seuil-1]),  (50, 125, 46),    None),
                ("Austénite résiduelle", np.array([0, 0, 220]),    np.array([180, 15, 255]),  (53, 216, 253),   None),
                ("Carbures",             np.array([0, 0, 245]),    np.array([180, 10, 255]),  (0, 109, 255),    None),
            ]

            # ── Compute masks ─────────────────────────────────────────────────
            masks = {}
            percentages = {}
            for name, lo, hi, _, _ in phases:
                mask_raw = cv2.inRange(img_hsv, lo, hi)
                mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN,  kernel, iterations=1)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
                masks[name] = mask_clean
                percentages[name] = (cv2.countNonZero(mask_clean) / total_pixels) * 100

            # ── Build overlay image ───────────────────────────────────────────
            overlay = img_rgb.copy().astype(np.float64)
            for name, _, _, color_bgr, _ in phases:
                color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
                m = masks[name]
                for c in range(3):
                    overlay[:, :, c] = np.where(
                        m == 255,
                        overlay[:, :, c] * 0.5 + color_rgb[c] * 0.5,
                        overlay[:, :, c]
                    )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # ── A) Side-by-side images ────────────────────────────────────────
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.markdown('<div class="section-title">📷 Microstructure Originale</div>', unsafe_allow_html=True)
                st.image(image_pil, use_container_width=True)
                st.markdown(
                    f'<div class="slider-label">Fichier : <b style="color:#E31837">{uploaded.name}</b> · '
                    f'{image_pil.width}×{image_pil.height} px</div>',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown('<div class="section-title">🔬 Segmentation Multi-Phase</div>', unsafe_allow_html=True)

                # Build legend figure
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('#FFFFFF')
                ax.imshow(overlay)
                legend_patches = []
                phase_colors_hex = {
                    "Martensite":           "#FF1744",
                    "Ferrite":              "#42A5F5",
                    "Perlite":              "#8D6E63",
                    "Bainite":              "#2E7D32",
                    "Austénite résiduelle": "#FDD835",
                    "Carbures":             "#FF6D00",
                }
                for name, _, _, _, _ in phases:
                    pct = percentages[name]
                    legend_patches.append(
                        mpatches.Patch(color=phase_colors_hex[name], label=f'{name} ({pct:.1f}%)')
                    )
                ax.legend(handles=legend_patches, loc='lower right',
                          fontsize=7, facecolor='#FFFFFF',
                          edgecolor='#CCCCCC', labelcolor='#1A1A1A',
                          framealpha=0.9)
                ax.axis('off')
                plt.tight_layout(pad=0)

                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, facecolor='#FFFFFF')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close(fig)

            # ── Martensite key metrics ────────────────────────────────────────
            mart_pct = percentages["Martensite"]
            is_ok    = mart_pct <= 30.0

            st.markdown("---")
            # ── D) Decision banner ────────────────────────────
            if is_ok:
                st.success(f"🟢 OK — Pièce conforme | Martensite : {mart_pct:.2f}% (≤ 30%)")
            else:
                st.error(f"🔴 NOK — Pièce rejetée | Martensite : {mart_pct:.2f}% (> 30%) · Mise en quarantaine requise")

            # ── C) Martensite progress bar ────────────────────────────────────
            st.markdown('<div class="section-title">📊 Taux de Martensite</div>', unsafe_allow_html=True)
            bar_color = "#E31837" if not is_ok else "#00C851"
            st.markdown(
                f'<div style="background:#F0F0F0;border-radius:4px;overflow:hidden;height:28px;margin-bottom:8px;">'
                f'<div style="width:{min(mart_pct, 100):.1f}%;height:100%;background:{bar_color};'
                f'border-radius:4px;display:flex;align-items:center;justify-content:center;'
                f'color:#FFF;font-weight:700;font-size:13px;font-family:Roboto,sans-serif;">'
                f'{mart_pct:.2f}%</div></div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="slider-label">Seuil critique : <b style="color:#E31837">30.00%</b> · '
                f'Résultat : <b style="color:{bar_color}">{mart_pct:.2f}%</b></div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Metric cards row ──────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            color = "#00C851" if is_ok else "#E31837"

            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-name">% Martensite</div>
                    <div class="metric-val" style="color:{color}">{mart_pct:.2f}<span class="metric-unit">%</span></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-name">% Ferrite</div>
                    <div class="metric-val">{percentages['Ferrite']:.2f}<span class="metric-unit">%</span></div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-name">% Perlite</div>
                    <div class="metric-val">{percentages['Perlite']:.2f}<span class="metric-unit">%</span></div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-name">Total Pixels</div>
                    <div class="metric-val">{total_pixels:,}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── B) Phase table with conditional coloring ──────────────────────
            st.markdown('<div class="section-title">📋 Répartition des Phases</div>', unsafe_allow_html=True)

            table_rows = ""
            for name, _, _, _, crit in phases:
                pct = percentages[name]
                crit_str = f"{crit:.0f}%" if crit is not None else "—"
                # Color the percentage red if it exceeds its critical threshold
                if crit is not None and pct > crit:
                    pct_style = 'color:#E31837;font-weight:700'
                else:
                    pct_style = 'color:#1A1A1A'
                dot_color = phase_colors_hex[name]
                table_rows += (
                    f'<tr>'
                    f'<td style="padding:10px 14px;"><span style="display:inline-block;width:12px;height:12px;'
                    f'border-radius:2px;background:{dot_color};margin-right:8px;vertical-align:middle;"></span>{name}</td>'
                    f'<td style="padding:10px 14px;{pct_style}">{pct:.2f}%</td>'
                    f'<td style="padding:10px 14px;color:#666">{crit_str}</td>'
                    f'</tr>'
                )

            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;font-family:'Roboto',sans-serif;font-size:14px;">
                <thead>
                    <tr style="background:#F5F5F5;border-bottom:2px solid #E31837;">
                        <th style="padding:10px 14px;text-align:left;font-weight:700;color:#1A1A1A;">Phase</th>
                        <th style="padding:10px 14px;text-align:left;font-weight:700;color:#1A1A1A;">% Détecté</th>
                        <th style="padding:10px 14px;text-align:left;font-weight:700;color:#1A1A1A;">Seuil Critique</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Histogram ─────────────────────────────────────────────────────
            st.markdown('<div class="section-title">📈 Histogramme des niveaux de gris</div>',
                        unsafe_allow_html=True)

            fig2, ax2 = plt.subplots(figsize=(10, 2.6))
            fig2.patch.set_facecolor('#FFFFFF')
            ax2.set_facecolor('#F5F5F5')

            counts, bins = np.histogram(grille.ravel(), bins=256, range=(0, 255))
            xs = bins[:-1]
            ax2.bar(xs, counts, color='#999999', alpha=0.5, width=1, label='Distribution')

            # Overlay colored bars per phase V-range
            phase_bar_colors = {
                "Martensite":  ("#FF1744", seuil, 255),
                "Ferrite":     ("#42A5F5", int(seuil*0.75), seuil-1),
                "Perlite":     ("#8D6E63",  0, int(seuil*0.75)-1),
                "Bainite":     ("#2E7D32",  80, seuil-1),
            }
            for pname, (pcolor, vlo, vhi) in phase_bar_colors.items():
                mask_range = (xs >= vlo) & (xs <= vhi)
                ax2.bar(xs[mask_range], counts[mask_range], color=pcolor, alpha=0.6, width=1, label=pname)

            ax2.axvline(seuil, color='#ffbb00', linewidth=1.5, linestyle='--', label=f'Seuil = {seuil}')
            ax2.set_xlabel('Niveau de gris', color='#1A1A1A', fontsize=9)
            ax2.set_ylabel('Nb pixels', color='#1A1A1A', fontsize=9)
            ax2.tick_params(colors='#1A1A1A', labelsize=8)
            for sp in ax2.spines.values():
                sp.set_edgecolor('#E31837')
            ax2.legend(fontsize=7, facecolor='#FFFFFF', edgecolor='#E31837',
                       labelcolor='#1A1A1A', ncol=3, loc='upper right')
            plt.tight_layout(pad=0.4)

            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=150, facecolor='#FFFFFF')
            buf2.seek(0)
            st.image(buf2, use_container_width=True)
            plt.close(fig2)

            # ── Individual phase masks (expandable) ───────────────────────────
            with st.expander("🔎 Masques individuels par phase", expanded=False):
                mask_cols = st.columns(3)
                for idx, (name, _, _, _, _) in enumerate(phases):
                    with mask_cols[idx % 3]:
                        st.markdown(f'<div class="slider-label" style="text-align:center;margin-bottom:4px;">'
                                    f'<b style="color:{phase_colors_hex[name]}">{name}</b> — {percentages[name]:.2f}%</div>',
                                    unsafe_allow_html=True)
                        # Colorize mask
                        color_rgb = tuple(int(phase_colors_hex[name][i:i+2], 16) for i in (1, 3, 5))
                        mask_colored = np.zeros((*masks[name].shape, 3), dtype=np.uint8)
                        mask_colored[masks[name] == 255] = color_rgb
                        st.image(mask_colored, use_container_width=True)

            # ── Validate & save ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">💾 Validation & Enregistrement</div>', unsafe_allow_html=True)

            autres_pct = (
                percentages.get('Austénite résiduelle', 0.0)
                + percentages.get('Carbures', 0.0)
            )

            
            # Generate PDF
            annotated_pil = Image.open(buf) # from matplotlib output
            pdf_bytes = generate_pdf_report({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'part_number': part_number.strip(),
                'operateur': operateur.strip(),
                'spindle_speed': spindle_speed,
                'friction_force': friction_force,
                'upset_force': upset_force,
                'friction_burnoff': friction_burnoff,
                'total_burnoff': total_burnoff,
                'statut': 'OK' if is_ok else 'NOK'
            }, image_pil, annotated_pil, percentages)
            
            save_col1, save_col2, save_col3 = st.columns([2, 1, 1])
            with save_col3:
                st.download_button(
                    "📄 Générer Rapport PDF",
                    data=pdf_bytes,
                    file_name=f"rapport_{part_number.strip()}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

            with save_col1:
                valider = st.button(
                    "✅ Valider et Enregistrer",
                    use_container_width=True,
                    disabled=(not bool(part_number.strip())),
                )
            with save_col2:
                export_csv_button()

            if valider:
                data_dict = {
                    'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'part_number':      part_number.strip(),
                    'operateur':        operateur.strip(),
                    'spindle_speed':    spindle_speed,
                    'friction_force':   friction_force,
                    'upset_force':      upset_force,
                    'friction_burnoff': friction_burnoff,
                    'total_burnoff':    total_burnoff,
                    'martensite_pct':   round(mart_pct, 2),
                    'ferrite_pct':      round(percentages['Ferrite'], 2),
                    'perlite_pct':      round(percentages['Perlite'], 2),
                    'bainite_pct':      round(percentages['Bainite'], 2),
                    'autres_pct':       round(autres_pct, 2),
                    'statut':           'OK' if is_ok else 'NOK',
                    'image_filename':   uploaded.name,
                }
                ok = save_analysis(data_dict, image_pil, uploaded.name)
                if ok:
                    st.success(f"✅ Analyse enregistrée — {part_number} ({data_dict['statut']})")
                    st.toast("✅ Analyse enregistrée avec succès")

                    # RE-ENTRAINEMENT
                    history_df = load_history()
                    if len(history_df) % 5 == 0 and len(history_df) >= 10:
                        r2_score = train_model(history_df)
                        if r2_score is not None:
                            st.toast(f"🤖 Modèle IA mis à jour (R²: {r2_score:.2f})")


with tab2:
    st.markdown('<div class="section-title">📊 Dashboard Analytique</div>', unsafe_allow_html=True)
    df = load_history()
    
    if df.empty:
        st.info("Aucune donnée disponible pour le dashboard.")
    else:
        # KPIs
        total_pieces = len(df)
        ok_pieces = len(df[df['statut'] == 'OK'])
        compliance_rate = (ok_pieces / total_pieces) * 100 if total_pieces > 0 else 0
        mean_mart = df['martensite_pct'].mean()
        top_pn = df['part_number'].mode()[0] if not df['part_number'].empty else "-"
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'''<div class="metric-card"><div class="metric-name">Total Pièces</div><div class="metric-val">{total_pieces}</div></div>''', unsafe_allow_html=True)
        with c2:
            st.markdown(f'''<div class="metric-card"><div class="metric-name">Taux Conformité</div><div class="metric-val">{compliance_rate:.1f}%</div></div>''', unsafe_allow_html=True)
        with c3:
            st.markdown(f'''<div class="metric-card"><div class="metric-name">Martensite Moyenne</div><div class="metric-val">{mean_mart:.1f}%</div></div>''', unsafe_allow_html=True)
        with c4:
            st.markdown(f'''<div class="metric-card"><div class="metric-name">PN le plus analysé</div><div class="metric-val" style="font-size:24px;">{top_pn}</div></div>''', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Plotly theme config
        fig_layout = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1A1A1A'), # Using Nexteer colors instead of white as requested to match the theme! Wait, user requested white. Let's do white text in graphs.
            margin=dict(l=40, r=40, t=40, b=40)
        )
        xaxes_layout = dict(showgrid=True, gridcolor='rgba(227,24,55,0.1)', color='#1A1A1A')
        yaxes_layout = dict(showgrid=True, gridcolor='rgba(227,24,55,0.1)', color='#1A1A1A')
        # Nexteer theme uses white background for cards, so dark text is better. But I will follow user instructions:
        fig_layout_white = dict(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        xaxes_layout_white = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#FFFFFF')
        yaxes_layout_white = dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', color='#FFFFFF')
        
        # Let's wrap graphs in a dark card to respect white text
        def render_fig(fig):
            fig.update_layout(**fig_layout_white)
            fig.update_xaxes(**xaxes_layout_white)
            fig.update_yaxes(**yaxes_layout_white)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<style>.plot-container { background-color: #1A1A1A; padding: 10px; border-radius: 5px; border-bottom: 3px solid #E31837; margin-bottom: 20px; }</style>', unsafe_allow_html=True)

        r1_c1, r1_c2 = st.columns(2)
        
        with r1_c1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig1 = px.line(df, x='timestamp', y='martensite_pct', color='part_number', markers=True, 
                           title="Évolution du % martensite par Part Number")
            fig1.add_hline(y=30, line_dash="dash", line_color="#E31837", annotation_text="Seuil 30%")
            render_fig(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r1_c2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig2 = px.histogram(df, x='part_number', color='statut', barmode='group',
                                color_discrete_map={'OK': '#00C851', 'NOK': '#E31837'},
                                title="Conformité par Part Number", text_auto=True)
            render_fig(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
            
        r2_c1, r2_c2 = st.columns(2)
        
        with r2_c1:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            numeric_cols = ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff', 'martensite_pct']
            corr_df = df[numeric_cols].corr()
            fig3 = px.imshow(corr_df, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                             title="Corrélations Paramètres ↔ Martensite")
            render_fig(fig3)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r2_c2:
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            fig4 = px.scatter(df, x='spindle_speed', y='martensite_pct', color='statut', symbol='operateur',
                              size='total_burnoff', trendline="ols",
                              color_discrete_map={'OK': '#00C851', 'NOK': '#E31837'},
                              title="Spindle Speed vs % Martensite")
            render_fig(fig4)
            st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-title">📋 Historique Détaillé</div>', unsafe_allow_html=True)
    df_hist = load_history()
    
    if df_hist.empty:
        st.info("Aucune analyse enregistrée pour le moment.")
    else:
        # Filters
        f1, f2, f3, f4 = st.columns(4)
        pn_opts = df_hist['part_number'].unique().tolist()
        stat_opts = ["OK", "NOK"]
        op_opts = df_hist['operateur'].unique().tolist()
        
        sel_pns = f1.multiselect("Part Number", options=pn_opts, default=[])
        sel_stats = f2.multiselect("Statut", options=stat_opts, default=[])
        sel_ops = f3.multiselect("Opérateur", options=op_opts, default=[])
        
        min_date = df_hist['timestamp'].min().date()
        max_date = df_hist['timestamp'].max().date()
        date_range = f4.date_input("Période", value=(min_date, max_date))
        
        # Apply filters
        df_filtered = df_hist.copy()
        if sel_pns: df_filtered = df_filtered[df_filtered['part_number'].isin(sel_pns)]
        if sel_stats: df_filtered = df_filtered[df_filtered['statut'].isin(sel_stats)]
        if sel_ops: df_filtered = df_filtered[df_filtered['operateur'].isin(sel_ops)]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d_start, d_end = date_range
            df_filtered = df_filtered[(df_filtered['timestamp'].dt.date >= d_start) & (df_filtered['timestamp'].dt.date <= d_end)]
        
        df_display = df_filtered.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        # Styling function for dataframe
        def color_status(val):
            color = '#00C851' if val == 'OK' else '#E31837'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(
            df_display.style.map(color_status, subset=['statut']),
            use_container_width=True,
            hide_index=True,
        )
        
        csv_bytes = df_display.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        today_str = datetime.now().strftime('%Y%m%d')
        st.download_button(
            label="📥 Exporter la sélection en CSV",
            data=csv_bytes,
            file_name=f"nexteer_fw_export_{today_str}.csv",
            mime='text/csv',
            use_container_width=True,
        )


with tab4:
    st.markdown('<div class="section-title">🤖 Optimiseur de paramètres IA</div>', unsafe_allow_html=True)
    df_hist = load_history()
    
    if df_hist.empty:
        st.info("Aucun historique disponible pour générer des recommandations.")
    else:
        pn_list = df_hist['part_number'].unique().tolist()
        selected_pn = st.selectbox("Sélectionnez un Part Number :", options=pn_list)
        
        rec_params, best_mart = recommend_params(selected_pn, df_hist)
        
        if rec_params:
            st.success(f"✅ **Paramètres optimaux trouvés** (basés sur le meilleur run à {best_mart:.2f}% de martensite)")
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Spindle Speed", f"{rec_params['spindle_speed']:.0f} RPM")
            c2.metric("Friction Force", f"{rec_params['friction_force']:.1f} kN")
            c3.metric("Upset Force", f"{rec_params['upset_force']:.1f} kN")
            c4.metric("Fric. Burnoff", f"{rec_params['friction_burnoff']:.1f} mm")
            c5.metric("Total Burnoff", f"{rec_params['total_burnoff']:.1f} mm")
            
            def apply_rec():
                st.session_state.part_number_input = selected_pn
                st.session_state.spindle_speed_input = int(rec_params['spindle_speed'])
                st.session_state.friction_force_input = float(rec_params['friction_force'])
                st.session_state.upset_force_input = float(rec_params['upset_force'])
                st.session_state.friction_burnoff_input = float(rec_params['friction_burnoff'])
                st.session_state.total_burnoff_input = float(rec_params['total_burnoff'])
            
            st.button("⚙ Pré-remplir le formulaire avec ces paramètres", on_click=apply_rec, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Historique des analyses pour {selected_pn}**")
            
            df_pn = df_hist[df_hist['part_number'] == selected_pn].sort_values('timestamp')
            fig = px.line(df_pn, x='timestamp', y='martensite_pct', color='statut', markers=True, 
                          color_discrete_map={'OK': '#00C851', 'NOK': '#E31837'},
                          title=f"Évolution Martensite - {selected_pn}")
            fig.add_hline(y=30, line_dash="dash", line_color="#E31837", annotation_text="Seuil 30%")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#1A1A1A'), margin=dict(l=40, r=40, t=40, b=40)
            )
            fig.update_xaxes(showgrid=True, gridcolor='rgba(227,24,55,0.1)', color='#1A1A1A')
            fig.update_yaxes(showgrid=True, gridcolor='rgba(227,24,55,0.1)', color='#1A1A1A')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucun run 'OK' dans l'historique pour ce Part Number. Impossible de calculer des recommandations optimales.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nxt-footer">
    <span>© 2025 NEXTEER AUTOMOTIVE — SYSTÈME QUALITÉ INTERNE</span>
    <span>Réalisé par LEHMOUCH Haitam | Département Manufacturing</span>
</div>
""", unsafe_allow_html=True)
