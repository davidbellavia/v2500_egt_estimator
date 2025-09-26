# app.py — V2500 EGTm Estimator (Keras + MinMaxScaler preprocessor) with tinted background
import os
import sys
import base64
import numpy as np
import pandas as pd
import streamlit as st

# Optional deps (only used if present)
try:
    import joblib
except Exception:
    joblib = None

try:
    import tensorflow as tf
except Exception:
    tf = None

# ---------- Page setup ----------
st.set_page_config(
    page_title="V2500 EGTm Estimator",
    # page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# (Optional) build banner so you can confirm the deployed file updated
import time
st.caption(f"🔧 Build {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ---------- Background image helper (with color tints + filters) ----------
def set_background(
    image_path: str,
    overlay_rgba: str = "rgba(255,255,255,0.66)",  # readability layer on top
    *,
    contrast: float = 1.35,
    brightness: float = 0.90,
    saturate: float = 1.18,
    red_boost: float = 0.20,    # 0.00–0.40 typical
    blue_boost: float = 0.25,   # 0.00–0.40 typical
    blend_mode: str = "overlay" # try: "overlay", "soft-light", "multiply", "screen"
):
    """
    Renders a full-page background by stacking:
      1) red tint layer
      2) blue tint layer
      3) the image
    then applies CSS filters and a top readability overlay.
    Also defines layout helpers: .app-title, .nowrap, .center, .hero, .tagline.
    """
    if not os.path.exists(image_path):
        st.warning(f"Background image not found at {image_path}")
        return

    b64 = base64.b64encode(open(image_path, "rb").read()).decode()

    st.markdown(
        f"""
        <style>
        /* Image & tints on ::before (behind content) */
        [data-testid="stAppViewContainer"]::before {{
            content: "";
            position: fixed; inset: 0;
            background-image:
                linear-gradient(rgba(255,0,0,{red_boost}), rgba(255,0,0,{red_boost})),
                linear-gradient(rgba(0,0,255,{blue_boost}), rgba(0,0,255,{blue_boost})),
                url("data:image/png;base64,{b64}");
            background-size: cover, cover, cover;
            background-position: center, center, center;
            background-attachment: fixed, fixed, fixed;
            background-blend-mode: {blend_mode}, {blend_mode}, normal;
            filter: contrast({contrast}) brightness({brightness}) saturate({saturate});
            z-index: 0;
        }}

        /* Readability overlay on ::after */
        [data-testid="stAppViewContainer"]::after {{
            content: ""; position: fixed; inset: 0;
            background: {overlay_rgba};
            z-index: 1;
        }}

        /* Keep main content above */
        [data-testid="stAppViewContainer"] > .main {{
            position: relative; z-index: 2; background: transparent;
        }}

        /* Transparent top header */
        [data-testid="stHeader"] {{ background: transparent; }}

        /* Sidebar styling (slightly opaque panel) */
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(255,255,255,0.88);
            backdrop-filter: saturate(1.2);
        }}

        /* Title single-line */
        .app-title {{
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0;
        }}

        /* Utilities */
        .nowrap {{ white-space: nowrap; }}
        .center {{ text-align: center; }}

        /* Centered stack for title + tagline */
        .hero {{ display:flex; flex-direction:column; align-items:center; gap:.15rem; }}
        .tagline {{ white-space:nowrap; text-align:center; margin:.25rem 0 .75rem 0; font-size:0.95rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Set the background (expects assets/united_787.png). Adjust params to taste.
set_background(
    "assets/united_787.png",
    overlay_rgba="rgba(255,255,255,0.66)",
    contrast=1.35,
    brightness=0.90,
    saturate=1.18,
    red_boost=0.20,
    blue_boost=0.25,
    blend_mode="overlay",
)

# ---------- Title + centered single-line instruction ----------
st.markdown(
    """
    <div class="hero">
      <h1 class="app-title">V2500 EGTm Estimator</h1>
      <p class="tagline">
        Enter FANSTD, LPCSTD, HPCSTD, HPTSTD, LPTSTD (positive, 1 decimal).
        Click <strong>Estimate EGTm</strong> to predict using your trained Keras model.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

FEATURES = ["FANSTD", "LPCSTD", "HPCSTD", "HPTSTD", "LPTSTD"]

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    """
    Loads:
      - ColumnTransformer/Scaler from preprocessor.joblib (or ct.joblib) fitted on training data
      - Keras model from model_1.keras
    Returns (preprocessor, model)
    """
    # Load preprocessor
    pre = None
    pre_candidates = ["preprocessor.joblib", "ct.joblib"]
    load_errs = []
    if joblib is not None:
        for fname in pre_candidates:
            if os.path.exists(fname):
                try:
                    pre = joblib.load(fname)
                    break
                except Exception as e:
                    load_errs.append(f"{fname}: {e}")
    else:
        load_errs.append("joblib not available; cannot load preprocessor.joblib/ct.joblib")

    # Load keras model
    mdl = None
    if tf is None:
        st.error("TensorFlow is not installed/available in this environment.")
    else:
        if os.path.exists("model_1.keras"):
            try:
                mdl = tf.keras.models.load_model("model_1.keras")
            except Exception as e:
                st.error(f"Failed to load model_1.keras: {e}")
        else:
            st.error("model_1.keras not found in the app directory.")

    # Helpful warnings
    if pre is None:
        st.warning(
            "Preprocessor (MinMaxScaler via ColumnTransformer) not found. "
            "Please save your fitted transformer as **preprocessor.joblib** (or ct.joblib) "
            "in the app folder. Example after training:\n\n"
            "```python\n"
            "import joblib\n"
            "joblib.dump(ct, 'preprocessor.joblib')\n"
            "model_1.save('model_1.keras')\n"
            "```",
        )
    if load_errs:
        st.caption("Preprocessor load notes:\n- " + "\n- ".join(load_errs))

    return pre, mdl

pre, model = load_artifacts()

# ---------- Helpers ----------
def prep_for_model(df: pd.DataFrame) -> np.ndarray:
    """
    Applies the fitted ColumnTransformer/MinMaxScaler (ct) exactly as in training.
    Converts to dense float32 if needed.
    """
    if pre is None:
        raise RuntimeError("Preprocessor not loaded. Provide preprocessor.joblib/ct.joblib.")
    X = pre.transform(df)
    if hasattr(X, "toarray"):  # scipy sparse -> dense
        X = X.toarray()
    return X.astype("float32", copy=False)

def _clean_pos_1dp(x: float) -> float:
    """Clip to >=0 and round to 1 decimal."""
    if x is None:
        return 0.0
    return round(max(0.0, float(x)), 1)

# ---------- Spacer to push controls low on the page ----------
# Adjust the vh value to move the row higher/lower (bigger = lower on page)
st.markdown("<div style='height:55vh'></div>", unsafe_allow_html=True)

# ---------- Single-line input row near bottom ----------
with st.form("egtm_form", clear_on_submit=False):
    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 1])

    with c1:
        FANSTD = st.number_input("FANSTD", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    with c2:
        LPCSTD = st.number_input("LPCSTD", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    with c3:
        HPCSTD = st.number_input("HPCSTD", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    with c4:
        HPTSTD = st.number_input("HPTSTD", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    with c5:
        LPTSTD = st.number_input("LPTSTD", min_value=0.0, value=0.0, step=0.1, format="%.1f")
    with c6:
        submitted = st.form_submit_button("Estimate EGTm", use_container_width=True)

# ---------- Predict single ----------
if submitted:
    try:
        if model is None:
            raise RuntimeError("Keras model not loaded. Place model_1.keras beside app.py.")

        # Enforce positivity & 1 decimal
        FANSTD_c = _clean_pos_1dp(FANSTD)
        LPCSTD_c = _clean_pos_1dp(LPCSTD)
        HPCSTD_c = _clean_pos_1dp(HPCSTD)
        HPTSTD_c = _clean_pos_1dp(HPTSTD)
        LPTSTD_c = _clean_pos_1dp(LPTSTD)

        row = pd.DataFrame(
            [{
                "FANSTD": FANSTD_c,
                "LPCSTD": LPCSTD_c,
                "HPCSTD": HPCSTD_c,
                "HPTSTD": HPTSTD_c,
                "LPTSTD": LPTSTD_c,
            }],
            columns=FEATURES
        )

        X1 = prep_for_model(row)
        pred_float = float(model.predict(X1, verbose=0).ravel()[0])
        pred_int = int(np.rint(pred_float))  # integer EGTm
        st.success(f"Estimated EGTm: **{pred_int:,d}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------- Footer ----------
st.caption(f"Runtime: Python {sys.version.split()[0]} | TensorFlow: {getattr(tf, '__version__', 'n/a')} | File: {__file__}")
