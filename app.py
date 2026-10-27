# =============================================================================
# AgroEspacio Colombia – Aplicación Streamlit (experiencia inmersiva)
# SpVAR + ESDA + mapas + 3D | Datos EVA reales + Interpretabilidad Automática
# =============================================================================

import sys, subprocess, importlib, warnings, zipfile, io, requests, tempfile, os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import inv, pinv
from scipy.spatial.distance import pdist, squareform
from libpysal.weights import Queen, Rook, KNN, DistanceBand, lag_spatial
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error, GM_Combo
import statsmodels.api as sm
import folium
from splot.esda import lisa_cluster, moran_scatterplot
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroEspacio Colombia",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para simular la experiencia inmersiva
st.markdown("""
<style>
    .hero {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #0f766e, #134e4a);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { font-size: 2.2rem; margin-bottom: 0.3rem; }
    .hero p { opacity: 0.92; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ESTADO GLOBAL (SESSION STATE)
# -----------------------------------------------------------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False
    st.session_state.gdf_static = None
    st.session_state.panel = None
    st.session_state.gdf_avg = None
    st.session_state.W_opt = None
    st.session_state.W_name = None
    st.session_state.panel_sp = None
    st.session_state.res1 = None
    st.session_state.res2 = None
    st.session_state.spec1 = None
    st.session_state.spec2 = None
    st.session_state.depts = None
    st.session_state.auto_insight = ""

# -----------------------------------------------------------------------------
# CARGA DE DATOS Y PIPELINE EMPÍRICO EVA
# -----------------------------------------------------------------------------
def load_pipeline():
    SHP_DIR = Path("/content/gadm_col")
    SHP_DIR.mkdir(exist_ok=True, parents=True)
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_COL_shp.zip"
    if not list(SHP_DIR.glob("*_1.shp")):
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(SHP_DIR)
    
    gdf = gpd.read_file(list(SHP_DIR.glob("*_1.shp"))[0])
    gdf = gdf.rename(columns={"NAME_1": "departamento", "GID_1": "dept_id"})
    gdf["departamento"] = (
        gdf["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    gdf = gdf[["dept_id", "departamento", "geometry"]].drop_duplicates("departamento")

    def fetch_eva(u, limit=50000):
        frames, offset = [], 0
        while True:
            resp = requests.get(u, params={"$limit": limit, "$offset": offset}, timeout=120)
            resp.raise_for_status()
            chunk = resp.json()
            if not chunk:
                break
            frames.append(pd.DataFrame(chunk))
            if len(chunk) < limit:
                break
            offset += limit
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    raw = pd.concat([
        fetch_eva("https://www.datos.gov.co/resource/2pnw-mmge.json"),
        fetch_eva("https://www.datos.gov.co/resource/uejq-wxrr.json"),
    ], ignore_index=True)

    cols = list(raw.columns)
    c_dept = next((c for c in cols if "departamento" in c.lower()
                   and "codigo" not in c.lower() and "c_digo" not in c.lower()), None)
    c_year = next((c for c in cols if c.lower().replace("_", "") in
                   ("ao", "ano", "anio", "año", "year", "a_o") or "a_o" in c.lower()), None)
    c_prod = next((c for c in cols if "producci" in c.lower()), None)
    c_area = next((c for c in cols if "sembrada" in c.lower()), None)
    
    eva = pd.DataFrame({
        "departamento": raw[c_dept].astype(str),
        "year": raw[c_year],
        "produccion_t": raw[c_prod],
        "area_sembrada": raw[c_area],
    })
    eva["departamento"] = (
        eva["departamento"].str.upper()
        .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
    )
    for c in ["year", "produccion_t", "area_sembrada"]:
        eva[c] = pd.to_numeric(eva[c], errors="coerce")
    eva["produccion_t"] = eva["produccion_t"].fillna(0)
    eva["area_sembrada"] = eva["area_sembrada"].fillna(0)
    eva = eva.dropna(subset=["departamento", "year"])

    panel = (
        eva.groupby(["departamento", "year"], as_index=False)
        .agg(agriculture=("produccion_t", "sum"), planted_area=("area_sembrada", "sum"))
    )
    panel = panel[panel["year"].between(2010, 2024)]
    panel = panel[(panel["agriculture"] > 0) | (panel["planted_area"] > 0)]
    panel["yield_tha"] = np.where(
        panel["planted_area"] > 0, panel["agriculture"] / panel["planted_area"], np.nan
    )

    depts_panel = sorted(panel["departamento"].unique())
    gdf = (
        gdf[gdf["departamento"].isin(depts_panel)]
        .drop_duplicates("departamento")
        .sort_values("departamento")
        .reset_index(drop=True)
    )
    depts = list(gdf["departamento"])
    panel = panel[panel["departamento"].isin(depts)].copy()

    avg = panel.groupby("departamento")[["agriculture", "planted_area", "yield_tha"]].mean().reindex(depts).reset_index()
    gdf_avg = gdf.merge(avg, on="departamento", how="left").fillna(0)

    candidates = {}
    try:
        candidates["Queen"] = Queen.from_dataframe(gdf, silence_warnings=True)
    except Exception:
        pass
    try:
        candidates["Rook"] = Rook.from_dataframe(gdf, silence_warnings=True)
    except Exception:
        pass
    for k in [3, 4, 5]:
        try:
            candidates[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, silence_warnings=True)
        except Exception:
            pass

    scores = {n: 0.0 for n in candidates}
    moran_results = {}
    for var in ["agriculture", "planted_area"]:
        y = gdf_avg[var].values
        for name, w in candidates.items():
            w.transform = "r"
            if len(y) != w.n:
                continue
            try:
                mi = Moran(y, w, permutations=199)
                scores[name] += abs(mi.I) * (1.0 if mi.p_sim < 0.05 else 0.25)
                if var == "agriculture":
                    moran_results[name] = mi
            except Exception:
                pass
    W_name = max(scores, key=scores.get)
    W_opt = candidates[W_name]
    W_opt.transform = "r"

    rows = []
    for yr in sorted(panel["year"].unique()):
        sub = panel[panel["year"] == yr].set_index("departamento")
        if not all(d in sub.index for d in depts):
            continue
        ya = np.array([sub.loc[d, "agriculture"] for d in depts])
        yp = np.array([sub.loc[d, "planted_area"] for d in depts])
        Wa, Wp = lag_spatial(W_opt, ya), lag_spatial(W_opt, yp)
        for i, d in enumerate(depts):
            rows.append({
                "departamento": d, "year": int(yr),
                "agriculture": ya[i], "planted_area": yp[i],
                "W_agriculture": Wa[i], "W_planted_area": Wp[i],
            })
    sp = pd.DataFrame(rows).sort_values(["departamento", "year"])
    for v in ["agriculture", "planted_area", "W_agriculture", "W_planted_area"]:
        sp[f"{v}_l1"] = sp.groupby("departamento")[v].shift(1)
    panel_sp = sp.dropna().reset_index(drop=True)

    def select_spec(y, X, W, name):
        results = {}
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        for label, builder in [
            ("OLS", lambda: OLS(y, X, name_y=name)),
            ("SAR", lambda: ML_Lag(y, X, W, name_y=name)),
            ("SEM", lambda: ML_Error(y, X, W, name_y=name)),
        ]:
            try:
                m = builder()
                results[label] = dict(
                    aic=m.aic, rho=getattr(m, "rho", None),
                    params=m.betas.flatten(), se=getattr(m, "std_err", None), spec=label
                )
            except Exception:
                pass
        if not results:
            raise RuntimeError("Sin modelos estimables")
        best = min(results.items(), key=lambda kv: kv[1]["aic"] if not np.isnan(kv[1]["aic"]) else 1e9)
        return best[0], best[1]

    y1 = panel_sp["agriculture"].values
    X1 = sm.add_constant(panel_sp[[
        "agriculture_l1", "planted_area_l1", "W_agriculture_l1", "W_planted_area_l1"
    ]].values)
    y2 = panel_sp["planted_area"].values
    X2 = sm.add_constant(panel_sp[[
        "planted_area_l1", "agriculture_l1", "W_planted_area_l1", "W_agriculture_l1"
    ]].values)
    spec1, res1 = select_spec(y1, X1, W_opt, "Agri")
    spec2, res2 = select_spec(y2, X2, W_opt, "Area")

    mi_best = moran_results.get(W_name, Moran(gdf_avg["agriculture"].values, W_opt, permutations=199))
    auto_insight = f"""
### 🧠 Interpretabilidad Automática & Diagnóstico Territorial
- **Efecto Espacial Global (Moran I):** `{mi_best.I:.4f}` (p-valor: `{mi_best.p_sim:.4f}`). 
  *Interpretación:* {'Existe autocorrelación espacial positiva fuerte; la producción agrícola tiende a concentrarse en clusters regionales.' if mi_best.p_sim < 0.05 else 'Distribución sin fuerte clustering global dominante.'}
- **Dinámica SpVAR Óptima:**
  - Producción modelada mediante **{spec1}**
  - Área sembrada modelada mediante **{spec2}**
  - Matriz de vecindad seleccionada: **{W_name}**
- **Implicación de Negocio:** Las políticas de incentivo a la producción y extensión agrícola no deben aislarse por departamento, ya que los choques se propagan entre vecinos a través de la red espacial.
    """

    st.session_state.gdf_static = gdf
    st.session_state.panel = panel
    st.session_state.gdf_avg = gdf_avg
    st.session_state.W_opt = W_opt
    st.session_state.W_name = W_name
    st.session_state.panel_sp = panel_sp
    st.session_state.res1 = res1
    st.session_state.res2 = res2
    st.session_state.spec1 = spec1
    st.session_state.spec2 = spec2
    st.session_state.depts = depts
    st.session_state.ready = True
    st.session_state.auto_insight = auto_insight

# -----------------------------------------------------------------------------
# INTERFAZ PRINCIPAL STREAMLIT
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>AgroEspacio Colombia</h1>
  <p>De datos EVA abiertos a inteligencia territorial espacial — SpVAR · ESDA · 3D</p>
  <p style="font-size:0.9rem;margin-top:0.5rem">Demo inmersiva · datos observados · con interpretabilidad automática</p>
</div>
""", unsafe_allow_html=True)

# Barra lateral para controles de carga y filtros
with st.sidebar:
    st.header("⚙️ Configuración y Controles")
    if not st.session_state.ready:
        if st.button("🚀 Cargar datos EVA + calibrar modelo", type="primary"):
            with st.spinner("Procesando datos y calibrando modelos espaciales (2–5 min)..."):
                load_pipeline()
            st.rerun()
    else:
        st.success("✅ Modelo y datos cargados")
        if st.button("🔄 Recargar datos"):
            st.session_state.ready = False
            st.rerun()

    st.markdown("---")
    if st.session_state.ready:
        years = sorted(st.session_state.panel["year"].unique().astype(int).tolist())
        selected_year = st.selectbox("📅 Año", years, index=len(years)-1)
        selected_var = st.selectbox(
            "📊 Variable",
            ["Producción (t)", "Área sembrada (ha)", "Rendimiento (t/ha)"],
            index=0
        )
    else:
        selected_year = 2024
        selected_var = "Producción (t)"

# Contenido principal condicional
if not st.session_state.ready:
    st.info("👈 Presiona el botón **'Cargar datos EVA + calibrar modelo'** en la barra lateral para iniciar la aplicación.")
else:
    # Pestañas principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "① Explorar territorio", 
        "② Diagnóstico ESDA", 
        "③ Inmersión 3D", 
        "④ Decisión SpVAR"
    ])

    with tab1:
        st.markdown("### Mapa coroplético de alto espectro")
        st.markdown("Visualización cartográfica basada en cuantiles para identificar la distribución departamental.")
        
        panel, gdf = st.session_state.panel, st.session_state.gdf_static
        sub = panel[panel["year"] == int(selected_year)]
        g = gdf.merge(sub, on="departamento", how="left")
        col_map = {
            "Producción (t)": ("agriculture", "YlGn", "toneladas"),
            "Área sembrada (ha)": ("planted_area", "YlOrBr", "hectáreas"),
            "Rendimiento (t/ha)": ("yield_tha", "plasma", "t/ha"),
        }
        col, cmap, title_u = col_map[selected_var]
        g[col] = g[col].fillna(0)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        g.plot(
            column=col, cmap=cmap, scheme="quantiles", k=7,
            legend=True, ax=ax, edgecolor="black", linewidth=0.35,
            legend_kwds={"loc": "lower left", "title": title_u, "fontsize": 8},
        )
        ax.set_title(f"{selected_var} · {selected_year}", fontweight="bold", fontsize=13)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.markdown("### ESDA e Interpretación Automática")
        st.markdown("Análisis exploratorio de datos espaciales (Moran y LISA).")
        
        gdf_avg, W = st.session_state.gdf_avg, st.session_state.W_opt
        col_map = {
            "Producción (t)": "agriculture",
            "Área sembrada (ha)": "planted_area",
            "Rendimiento (t/ha)": "yield_tha",
        }
        col = col_map[selected_var]
        mi = Moran(gdf_avg[col].values, W, permutations=499)
        sig = "sí hay dependencia espacial significativa" if mi.p_sim < 0.05 else "no hay evidencia fuerte de dependencia"
        
        st.markdown(f"""
        - **Índice de Moran I** = `{mi.I:.4f}`
        - **p-valor simulado** = `{mi.p_sim:.4f}`
        - **Interpretación Automática:** {sig}.
        """)
        
        y = gdf_avg[col].values
        lisa = Moran_Local(y, W, permutations=499)
        fig, ax = plt.subplots(figsize=(8, 8))
        lisa_cluster(lisa, gdf_avg, p=0.05, ax=ax, legend=True)
        ax.set_title(f"LISA ESDA · {selected_var}\nHH rojo · LL azul · outliers HL/LH", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.markdown("### Grafología satelital-espacial 3D")
        st.markdown("Eje Z = producción · Color = rendimiento · Tamaño = área sembrada.")
        
        panel, gdf = st.session_state.panel, st.session_state.gdf_static
        g_3d = gdf.merge(panel[panel["year"] == int(selected_year)], on="departamento", how="left")
        g_3d["lon"] = g_3d.geometry.centroid.x
        g_3d["lat"] = g_3d.geometry.centroid.y
        g_3d = g_3d.dropna(subset=["agriculture", "planted_area"])
        
        fig3d = px.scatter_3d(
            g_3d, x="lon", y="lat", z="agriculture",
            color="yield_tha", size="planted_area",
            hover_name="departamento",
            color_continuous_scale="plasma",
            title=f"Grafología 3D · {selected_year}",
            labels={"agriculture": "Producción (t)", "yield_tha": "Rendimiento", "planted_area": "Área"},
        )
        fig3d.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with tab4:
        st.markdown("### Mecanismos de transmisión y Narrativa SpVAR")
        st.markdown("Resultados econométricos espaciales para la toma de decisiones estratégicas.")
        
        st.markdown(st.session_state.auto_insight)
        
        r1, r2 = st.session_state.res1, st.session_state.res2
        p1, p2 = r1["params"], r2["params"]
        rho1 = r1["rho"] if r1["rho"] is not None else 0.0
        rho2 = r2["rho"] if r2["rho"] is not None else 0.0
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"**Ecuación de Producción ({st.session_state.spec1})**")
            df_p1 = pd.DataFrame({
                "Mecanismo": ["Autodinámica", "Cruce local (área)", "Espacial directa W·prod", "Espacial cruzada W·área", "ρ contemporáneo"],
                "Coeficiente": [f"{p1[1]:.4f}", f"{p1[2]:.4f}", f"{p1[3]:.4f}", f"{p1[4]:.4f}", f"{rho1:.4f}"]
            })
            st.dataframe(df_p1, use_container_width=True)
            
        with col_res2:
            st.markdown(f"**Ecuación de Área ({st.session_state.spec2})**")
            df_p2 = pd.DataFrame({
                "Mecanismo": ["Autodinámica", "Cruce local (prod)", "Espacial directa W·área", "Espacial cruzada W·prod", "ρ contemporáneo"],
                "Coeficiente": [f"{p2[1]:.4f}", f"{p2[2]:.4f}", f"{p2[3]:.4f}", f"{p2[4]:.4f}", f"{rho2:.4f}"]
            })
            st.dataframe(df_p2, use_container_width=True)

st.markdown("---")
st.markdown("**Posicionamiento:** *AgroEspacio* transforma datos agropecuarios abiertos en evidencia espacial accionable.")
