from __future__ import annotations
import time
import math
import random
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.plant_fopdt import FOPDTPlant
from src.controller_v3 import DryerHumidityCtrlV3
from src.theilsen_slope import MinuteMeanBuilder, TheilSenSlopeEstimator
from src.utils import clamp

# -----------------------------
# UI / Theme
# -----------------------------
st.set_page_config(page_title="Simulador MPC Umidade — BlockFiel V3 (Python)", layout="wide")

CSS = """
<style>
  .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
  .card {
    border-radius: 16px; padding: 14px 16px; background: #0f172a;
    border: 1px solid rgba(148,163,184,0.25); box-shadow: 0 10px 24px rgba(0,0,0,0.18);
  }
  .card h3 { margin: 0; font-size: 14px; color: rgba(226,232,240,0.9); font-weight: 600; }
  .metric { font-size: 28px; font-weight: 800; color: #e2e8f0; line-height: 1.1; }
  .sub { font-size: 12px; color: rgba(226,232,240,0.7); }
  .pill {
    display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
    border: 1px solid rgba(148,163,184,0.35); color: rgba(226,232,240,0.85);
    background: rgba(2,6,23,0.4);
  }
  .tiny { font-size: 11px; color: rgba(226,232,240,0.7); }
  .ok { color: #22c55e; font-weight: 700; }
  .warn { color: #f59e0b; font-weight: 700; }
  .bad { color: #ef4444; font-weight: 700; }
  .hr { border-top: 1px solid rgba(148,163,184,0.22); margin: 10px 0 6px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Session state init
# -----------------------------
if "plant" not in st.session_state:
    st.session_state.plant = FOPDTPlant()
    st.session_state.plant.reset()

if "ctrl" not in st.session_state:
    st.session_state.ctrl = DryerHumidityCtrlV3()

if "mm" not in st.session_state:
    st.session_state.mm = MinuteMeanBuilder()

if "ts" not in st.session_state:
    st.session_state.ts = TheilSenSlopeEstimator(window_n=11, clip_abs=0.20)

if "t_sec" not in st.session_state:
    st.session_state.t_sec = 0.0

if "qcs_next_sec" not in st.session_state:
    st.session_state.qcs_next_sec = 0.0

if "pv_raw" not in st.session_state:
    st.session_state.pv_raw = st.session_state.plant.pv_true

if "slope" not in st.session_state:
    st.session_state.slope = 0.0

if "minute_means" not in st.session_state:
    st.session_state.minute_means = deque(maxlen=2000)  # (minute_idx, pv1min)

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=50000)  # dict rows

if "running" not in st.session_state:
    st.session_state.running = False

# -----------------------------
# Helpers
# -----------------------------
def card(title: str, value: str, sub: str = "", pill: str = ""):
    st.markdown(
        f"""<div class="card">
            <h3>{title} {f'<span class="pill">{pill}</span>' if pill else ''}</h3>
            <div class="metric">{value}</div>
            <div class="sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True
    )

def plot_timeseries(df: pd.DataFrame, x: str, series: list[tuple[str,str]], title: str):
    fig = go.Figure()
    for col, name in series:
        if col in df:
            fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=name))
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=55, r=20, t=45, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Tempo (min)"),
    )
    fig.update_yaxes(title="")
    return fig

def ensure_qcs_schedule(min_sec: float, max_sec: float):
    if st.session_state.qcs_next_sec <= 0:
        st.session_state.qcs_next_sec = random.uniform(min_sec, max_sec)

def qcs_maybe_sample(meas_noise: float, min_sec: float, max_sec: float):
    """Gera PV QCS irregular: atualiza pv_raw apenas quando chega amostra."""
    ensure_qcs_schedule(min_sec, max_sec)
    dt = st.session_state.params["Ts_exec_min"] * 60.0
    st.session_state.qcs_next_sec -= dt
    got = False
    if st.session_state.qcs_next_sec <= 0:
        pv_true = st.session_state.plant.pv_true
        st.session_state.pv_raw = pv_true + random.gauss(0.0, meas_noise)
        st.session_state.qcs_next_sec = random.uniform(min_sec, max_sec)
        got = True
    return got

def slope_update_if_needed(got_new_qcs: bool):
    """Atualiza PV_1min e slope Theil–Sen.
    Importante: o usuário pediu N endpoints = N últimas PV_1min (médias por minuto)."""
    if not got_new_qcs:
        return

    t_sec = st.session_state.t_sec
    pv = st.session_state.pv_raw

    finalized = st.session_state.mm.update(t_sec, pv)
    for minute_idx, pv1 in finalized:
        st.session_state.minute_means.append((minute_idx, pv1))
        s = st.session_state.ts.add_minute_mean(minute_idx, pv1)
        st.session_state.slope = s

def make_df():
    if not st.session_state.history:
        return pd.DataFrame(columns=["t_min"])
    df = pd.DataFrame(list(st.session_state.history))
    return df

# -----------------------------
# Sidebar: Config
# -----------------------------
st.sidebar.title("⚙️ Configuração do Simulador")
st.sidebar.caption("BlockFiel V3 (Elipse) + Slope Theil–Sen baseado em PV_1min (média por minuto).")

with st.sidebar.expander("⏱️ Execução & QCS", expanded=True):
    Ts_exec_sec = st.slider("Período de execução do bloco (s)", 5, 30, 10, 1)
    Ts_exec_min = Ts_exec_sec / 60.0

    qcs_min = st.slider("QCS: intervalo mínimo (s)", 10, 60, 15, 1)
    qcs_max = st.slider("QCS: intervalo máximo (s)", 15, 120, 35, 1)
    meas_noise = st.slider("Ruído de medição QCS (%, desvio padrão)", 0.0, 0.10, 0.01, 0.005)

with st.sidebar.expander("📈 Slope (Theil–Sen) — PV_1min", expanded=True):
    N_endpoints = st.slider("N endpoints (últimas N médias de minuto)", 3, 31, 11, 1)
    clip_slope = st.slider("Clip slope |%/min| (proteção)", 0.0, 1.0, 0.20, 0.01)

with st.sidebar.expander("🏭 Planta FOPDT (simulação)", expanded=False):
    pv0 = st.number_input("PV0 (%)", value=10.0, step=0.1)
    mv0 = st.number_input("MV0 (bar)", value=3.8, step=0.05)
    tau = st.number_input("Tau (min)", value=10.0, step=0.5, min_value=0.5)
    deadtime = st.number_input("Dead time real (min)", value=7.0, step=0.5, min_value=0.0)

    K_mv = st.number_input("Ganho MV→PV (%/bar) (NEGATIVO)", value=-0.08, step=0.005, format="%.3f")
    K_prod = st.number_input("Ganho PROD→PV (%/t/h)", value=0.002, step=0.0005, format="%.4f")
    K_broke = st.number_input("Ganho BROKE→PV (%/t/h)", value=0.010, step=0.001, format="%.3f")
    K_ph = st.number_input("Ganho pH→PV (%/pH)", value=0.050, step=0.005, format="%.3f")
    proc_noise = st.slider("Ruído de processo (%, desvio padrão)", 0.0, 0.10, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ MPC (DRYER_HUMIDITY_CTRL V3)")
with st.sidebar.expander("Setpoint / entradas", expanded=True):
    sp = st.number_input("SP Umidade (%)", value=10.00, step=0.01, format="%.2f")
    mv_track = st.number_input("MV Track (bar)", value=3.80, step=0.01, format="%.2f")
    en_auto = st.checkbox("AUTO (habilita controle)", value=True)
    en_ff = st.checkbox("Feedforward ativo", value=True)
    do_reset = st.checkbox("Reset (1 ciclo)", value=False)

    prod = st.number_input("Produção (t/h)", value=170.0, step=1.0)
    broke = st.number_input("Broke (t/h)", value=0.0, step=0.5)
    ph = st.number_input("pH água branca", value=7.0, step=0.05)

with st.sidebar.expander("Parâmetros principais", expanded=False):
    LAMBDA_min = st.slider("LAMBDA (min)", 0.0, 12.0, 3.0, 0.1)
    Kp_base = st.number_input("Kp_base (bar/step)", value=0.02, step=0.001, format="%.4f")
    Ti_base = st.number_input("Ti_base (min)", value=10.0, step=0.5)
    ERR_STEP = st.number_input("ERR_STEP (%, define 1 'step')", value=0.1, step=0.01, format="%.2f")
    GAMMA = st.slider("GAMMA_SLOPE_RAW (0=erro atual, 1=erro predito)", 0.0, 1.0, 0.5, 0.01)
    mvMin = st.number_input("MV Min (bar)", value=0.0, step=0.1)
    mvMax = st.number_input("MV Max (bar)", value=10.0, step=0.1)

with st.sidebar.expander("Feedforward (base)", expanded=False):
    PROD_REF = st.number_input("PROD_REF (t/h)", value=170.0, step=1.0)
    BROKE_REF = st.number_input("BROKE_REF (t/h)", value=0.0, step=0.5)
    PH_REF = st.number_input("PH_REF (pH)", value=7.0, step=0.05)

    Kff_Prod = st.number_input("Kff_Prod_base (bar/(t/h))", value=0.0, step=0.001, format="%.4f")
    Kff_Broke = st.number_input("Kff_Broke_base (bar/(t/h))", value=0.0, step=0.001, format="%.4f")
    Kff_pH = st.number_input("Kff_pH_base (bar/pH)", value=0.0, step=0.001, format="%.4f")

with st.sidebar.expander("PV Acceptance + AutoTune + Supervisor + AutoFF", expanded=False):
    PV_EPS_ACCEPT = st.number_input("PV_EPS_ACCEPT (%)", value=0.002, step=0.001, format="%.4f")
    PV_MAX_HOLD_sec = st.number_input("PV_MAX_HOLD_sec (s)", value=45.0, step=5.0)
    AT_SAT_MARGIN = st.number_input("AT_SAT_MARGIN (bar)", value=0.05, step=0.01, format="%.3f")

    TARGET_STD_PV = st.number_input("TARGET_STD_PV (%)", value=0.10, step=0.01, format="%.2f")
    DEADTIME_min = st.number_input("DEADTIME_min (min)", value=7.0, step=0.5)
    WIN_MULT_DT = st.number_input("WIN_MULT_DT", value=3.0, step=1.0)
    MAX_STEP_FRAC = st.number_input("MAX_STEP_FRAC", value=0.10, step=0.01, format="%.2f")
    Kp_min = st.number_input("Kp_min", value=0.002, step=0.001, format="%.4f")
    Kp_max = st.number_input("Kp_max", value=0.200, step=0.010, format="%.3f")
    AUTOTUNE_FAST_EN = st.checkbox("AUTOTUNE_FAST_EN (anti-oscilação)", value=True)

    KPI_WIN_min = st.number_input("KPI_WIN_min (min) (ex: 480=8h)", value=480.0, step=30.0)
    KPI_UPDATE_min = st.number_input("KPI_UPDATE_min (min)", value=60.0, step=5.0)
    KP_SUPERVISOR_EN = st.checkbox("KP_SUPERVISOR_EN (aumento lento)", value=True)
    KP_SUP_MAXSTEP_FRAC = st.number_input("KP_SUP_MAXSTEP_FRAC", value=0.03, step=0.01, format="%.2f")

    FF_AUTOTUNE_EN = st.checkbox("FF_AUTOTUNE_EN", value=False)
    FF_MAXSTEP_PROD = st.number_input("FF_MAXSTEP_PROD", value=0.05, step=0.01, format="%.2f")
    FF_MAXSTEP_BROKE = st.number_input("FF_MAXSTEP_BROKE", value=0.05, step=0.01, format="%.2f")
    FF_MAXSTEP_PH = st.number_input("FF_MAXSTEP_PH", value=0.05, step=0.01, format="%.2f")
    FF_RANGE_FRAC = st.number_input("FF_RANGE_FRAC (± faixa total)", value=1.0, step=0.5, format="%.1f")

# Consolidate params in session_state for reuse in simulation loop
st.session_state.params = {
    "Ts_exec_min": Ts_exec_min,
    "LAMBDA_min": LAMBDA_min,
    "Kp_base": Kp_base,
    "Ti_base_min": Ti_base,
    "MV_Min_bar": mvMin,
    "MV_Max_bar": mvMax,
    "Kff_Prod_base": Kff_Prod,
    "Kff_Broke_base": Kff_Broke,
    "Kff_pH_base": Kff_pH,
    "GAMMA_SLOPE_RAW": GAMMA,
    "ERR_STEP": ERR_STEP,
    "PV_EPS_ACCEPT": PV_EPS_ACCEPT,
    "PV_MAX_HOLD_sec": PV_MAX_HOLD_sec,
    "AT_SAT_MARGIN": AT_SAT_MARGIN,
    "AUTOTUNE_FAST_EN": AUTOTUNE_FAST_EN,
    "TARGET_STD_PV": TARGET_STD_PV,
    "DEADTIME_min": DEADTIME_min,
    "WIN_MULT_DT": WIN_MULT_DT,
    "MAX_STEP_FRAC": MAX_STEP_FRAC,
    "Kp_min": Kp_min,
    "Kp_max": Kp_max,
    "PROD_REF": PROD_REF,
    "BROKE_REF": BROKE_REF,
    "PH_REF": PH_REF,
    "KPI_WIN_min": KPI_WIN_min,
    "KPI_UPDATE_min": KPI_UPDATE_min,
    "KP_SUPERVISOR_EN": KP_SUPERVISOR_EN,
    "KP_SUP_MAXSTEP_FRAC": KP_SUP_MAXSTEP_FRAC,
    "FF_AUTOTUNE_EN": FF_AUTOTUNE_EN,
    "FF_MAXSTEP_PROD": FF_MAXSTEP_PROD,
    "FF_MAXSTEP_BROKE": FF_MAXSTEP_BROKE,
    "FF_MAXSTEP_PH": FF_MAXSTEP_PH,
    "FF_RANGE_FRAC": FF_RANGE_FRAC,
}

# Apply slope estimator settings
st.session_state.ts.set_window(N_endpoints)
st.session_state.ts.set_clip(clip_slope)

# Apply plant settings
plant = st.session_state.plant
plant.pv0 = pv0
plant.mv0 = mv0
plant.tau_min = tau
plant.deadtime_min = deadtime
plant.K_mv = K_mv
plant.K_prod = K_prod
plant.K_broke = K_broke
plant.K_ph = K_ph
plant.noise_std = proc_noise

# -----------------------------
# Main layout: Title + controls
# -----------------------------
st.title("Simulador MPC Umidade — Python (BlockFiel V3 + Slope Theil–Sen)")

colA, colB, colC, colD = st.columns([1.2, 1.2, 1.2, 1.4])
with colA:
    if st.button("▶ Play / Rodar", use_container_width=True):
        st.session_state.running = True
with colB:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.running = False
with colC:
    if st.button("⏭ Step (1 ciclo)", use_container_width=True):
        st.session_state.running = False
        st.session_state._do_one_step = True
with colD:
    if st.button("♻ Reset total", use_container_width=True):
        st.session_state.running = False
        st.session_state.t_sec = 0.0
        st.session_state.qcs_next_sec = 0.0
        st.session_state.history.clear()
        st.session_state.minute_means.clear()
        st.session_state.mm = MinuteMeanBuilder()
        st.session_state.ts = TheilSenSlopeEstimator(window_n=N_endpoints, clip_abs=clip_slope)
        st.session_state.plant.reset()
        st.session_state.pv_raw = st.session_state.plant.pv_true
        st.session_state.slope = 0.0
        st.session_state.ctrl = DryerHumidityCtrlV3()
        st.success("Reset completo aplicado.")

# Speed controls
c1, c2, c3 = st.columns([1,1,1])
with c1:
    steps_per_click = st.slider("Passos por execução (Play)", 1, 400, 80, 1)
with c2:
    sleep_ms = st.slider("Velocidade (sleep por passo, ms)", 0, 200, 20, 5)
with c3:
    window_min = st.slider("Janela de plot (min) — rolável", 10, 600, 120, 10)

# -----------------------------
# Simulation step function
# -----------------------------
ctrl = st.session_state.ctrl

def sim_step():
    # 1) planta
    dt_min = st.session_state.params["Ts_exec_min"]
    # MV usada na planta = último comando (se já existe), senão mv_track
    last_mv_cmd = st.session_state.history[0]["MV_CMD"] if st.session_state.history else mv_track
    pv_true = plant.step(dt_min, last_mv_cmd, prod, broke, ph)

    # 2) QCS (irregular)
    got_new_qcs = qcs_maybe_sample(meas_noise, qcs_min, qcs_max)

    # 3) slope (atualiza apenas quando chega QCS; PV_1min fecha ao virar minuto)
    slope_update_if_needed(got_new_qcs)

    # 4) controlador
    out = ctrl.step(
        inputs=dict(
            sp=sp, pv_raw=st.session_state.pv_raw, slope_raw=st.session_state.slope,
            mv_track=mv_track, prod=prod, broke=broke, ph=ph,
            en_auto=en_auto, en_ff=en_ff, do_reset=do_reset
        ),
        params=st.session_state.params
    )

    # 5) salva histórico
    st.session_state.t_sec += dt_min * 60.0
    t_min = st.session_state.t_sec / 60.0

    row = {
        "t_min": t_min,
        "PV_true": pv_true,
        "PV_raw": st.session_state.pv_raw,
        "PV_used": out["pv_used"],
        "SP": sp,
        "Slope": st.session_state.slope,
        "Slope_used": out["slope_used"],
        "PV_pred": out["PV_PRED"],
        "Err_now": out["ERR_NOW"],
        "Err_pred": out["ERR_PRED"],
        "Err_eff": out["e_eff"],
        "MV_CMD": out["MV_CMD"],
        "MV_FF": out["MV_FF"],
        "MV_FB": out["MV_FB"],
        "pTerm": out["pTerm"],
        "iTerm": out["iTerm"],
        "Kp": out["KP_EFF"],
        "Ti": out["TI_EFF"],
        "Kff_prod": out["KFF_PROD_EFF"],
        "Kff_broke": out["KFF_BROKE_EFF"],
        "Kff_ph": out["KFF_PH_EFF"],
        "PV_accepted": 1 if out["PV_ACCEPTED"] else 0,
        "DT_real": out["DT_REAL_MIN"],
        "KPI_stdPV": out["KPI_STD_PV"],
        "KPI_zc": out["KPI_ZC_RATIO"],
        "KPI_effort": out["KPI_EFFORT"],
        "NearSat": 1 if out["NEAR_SAT"] else 0
    }
    st.session_state.history.appendleft(row)

# Run simulation
if st.session_state.get("_do_one_step", False):
    st.session_state._do_one_step = False
    sim_step()

if st.session_state.running:
    placeholder = st.empty()
    for _ in range(steps_per_click):
        sim_step()
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    placeholder.empty()

# -----------------------------
# Build dataframe and apply view window
# -----------------------------
df = make_df()
if len(df) > 0:
    df = df.sort_values("t_min")
    tmax = df["t_min"].max()
    tmin = max(0.0, tmax - window_min)
    dfw = df[df["t_min"] >= tmin].copy()
else:
    dfw = df

# -----------------------------
# Faceplate (top cards)
# -----------------------------
top = st.container()
with top:
    cols = st.columns(6)
    if len(dfw) > 0:
        last = dfw.iloc[-1]
        pv_used = last["PV_used"]
        pv_raw_v = last["PV_raw"]
        pv_pred = last["PV_pred"]
        slope_v = last["Slope_used"]
        mv_cmd = last["MV_CMD"]
        kp_v = last["Kp"]
        ti_v = last["Ti"]
        stdpv = last["KPI_stdPV"]
    else:
        pv_used = st.session_state.pv_raw
        pv_raw_v = st.session_state.pv_raw
        pv_pred = st.session_state.pv_raw
        slope_v = st.session_state.slope
        mv_cmd = mv_track
        kp_v = ctrl.kp_eff
        ti_v = ctrl.ti_eff
        stdpv = ctrl.kpi_std_pv

    with cols[0]:
        card("PV usada no controle", f"{pv_used:.3f}%", sub="(PV Acceptance, IN2→PV)", pill="PV_used")
    with cols[1]:
        card("PV QCS (raw)", f"{pv_raw_v:.3f}%", sub="Amostragem irregular (15..35s etc.)", pill="PV_raw")
    with cols[2]:
        card("PV predita", f"{pv_pred:.3f}%", sub="PV + slope * LAMBDA", pill="PV_pred")
    with cols[3]:
        card("Slope (Theil–Sen)", f"{slope_v:+.4f} %/min", sub=f"N endpoints={N_endpoints} (últimas PV_1min)", pill="IN3")
    with cols[4]:
        card("MV_CMD", f"{mv_cmd:.3f} bar", sub="Comando total: FF + FB", pill="OUT1")
    with cols[5]:
        card("Kp / Ti", f"{kp_v:.4f} / {ti_v:.2f}", sub=f"KPI std(PV)={stdpv:.3f}%", pill="OUT7/OUT9")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📉 Slope PV_1min", "🛠 AutoTune & Logs", "🧮 Estático (1 ciclo)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = plot_timeseries(dfw, "t_min", [
            ("SP","SP"),
            ("PV_used","PV usada"),
            ("PV_raw","PV raw"),
            ("PV_pred","PV predita"),
        ], "Umidade (PV/SP/PV_pred)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = plot_timeseries(dfw, "t_min", [
            ("MV_CMD","MV_CMD"),
            ("MV_FF","MV_FF"),
            ("MV_FB","MV_FB"),
        ], "MV (componente total / FF / FB)")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = plot_timeseries(dfw, "t_min", [
            ("Err_now","Erro atual"),
            ("Err_pred","Erro predito"),
            ("Err_eff","Erro efetivo"),
        ], "Erros (%): atual / predito / efetivo")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = plot_timeseries(dfw, "t_min", [
            ("Kp","Kp_eff"),
            ("Ti","Ti_eff"),
        ], "Ganhos efetivos (Kp/Ti)")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Dica: use a janela de plot (min) para 'voltar no tempo' e analisar histórico com calma.")

with tab2:
    st.subheader("Slope virtual (Theil–Sen) com PV_1min e N endpoints")
    st.write("**Definição:** se N=5, o slope é calculado usando as **5 últimas médias de minuto** (PV_1min).")
    if st.session_state.minute_means:
        mm_df = pd.DataFrame(list(st.session_state.minute_means), columns=["minute", "PV_1min"]).sort_values("minute")
        mm_df["t_min"] = mm_df["minute"].astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mm_df["t_min"], y=mm_df["PV_1min"], mode="lines+markers", name="PV_1min (média do minuto)"))
        fig.update_layout(height=320, margin=dict(l=55,r=20,t=45,b=40), title="PV_1min (médias por minuto)")
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar endpoints usados (últimos N)
        used = mm_df.tail(N_endpoints).copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=used["t_min"], y=used["PV_1min"], mode="markers+lines", name=f"Últimos {N_endpoints} endpoints"))
        fig2.update_layout(height=260, margin=dict(l=55,r=20,t=45,b=40), title="Endpoints usados no Theil–Sen (janela)")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(used.tail(min(20, len(used))), use_container_width=True, height=240)
    else:
        st.info("Ainda não há PV_1min (aguarde alguns minutos de simulação).")

with tab3:
    st.subheader("AutoTune e Logs (fidelidade V3)")
    st.write("A camada FAST **só suaviza** (Kp↓ e/ou Ti↑) quando detecta oscilação/agressividade; "
             "o Supervisor pode aumentar Kp lentamente **somente se estável**; AutoFF é separado.")
    logs = list(ctrl.logs)[:40]
    if logs:
        for tag, msg in logs:
            st.markdown(f"- **{tag}**: {msg}")
    else:
        st.info("Sem eventos registrados ainda.")

    if len(dfw) > 0:
        fig = plot_timeseries(dfw, "t_min", [
            ("KPI_stdPV","KPI std(PV)"),
            ("KPI_zc","KPI zc ratio"),
            ("KPI_effort","KPI effort"),
        ], "KPIs do Supervisor")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Cálculo estático (1 ciclo)")
    st.caption("Simula **um** ciclo do bloco com os valores atuais, sem rodar a planta.")
    c1, c2 = st.columns(2)
    with c1:
        sp_s = st.number_input("SP (%)", value=float(sp), step=0.01, key="sp_static")
        pv_s = st.number_input("PV_raw (%)", value=float(st.session_state.pv_raw), step=0.01, key="pv_static")
        slope_s = st.number_input("Slope (%/min)", value=float(st.session_state.slope), step=0.001, format="%.4f", key="slope_static")
        prod_s = st.number_input("Prod (t/h)", value=float(prod), step=1.0, key="prod_static")
        broke_s = st.number_input("Broke (t/h)", value=float(broke), step=0.5, key="broke_static")
        ph_s = st.number_input("pH", value=float(ph), step=0.05, key="ph_static")
    with c2:
        en_auto_s = st.checkbox("AUTO", value=en_auto, key="auto_static")
        en_ff_s = st.checkbox("FF", value=en_ff, key="ff_static")
        mv_track_s = st.number_input("MV_track (bar)", value=float(mv_track), step=0.01, key="mvtrack_static")
        if st.button("Calcular 1 ciclo", use_container_width=True):
            tmp_ctrl = DryerHumidityCtrlV3()
            # inicializa com estados atuais para refletir integrador/ganhos
            tmp_ctrl.init_done = True
            tmp_ctrl.int_term = ctrl.int_term
            tmp_ctrl.prev_enable = ctrl.prev_enable
            tmp_ctrl.kp_eff = ctrl.kp_eff
            tmp_ctrl.ti_eff = ctrl.ti_eff
            tmp_ctrl.pv_last_acc = ctrl.pv_last_acc
            tmp_ctrl.slope_last_acc = ctrl.slope_last_acc
            tmp_ctrl.has_accepted_once = True
            tmp_ctrl.time_since_acc_min = ctrl.time_since_acc_min

            out = tmp_ctrl.step(
                inputs=dict(sp=sp_s, pv_raw=pv_s, slope_raw=slope_s, mv_track=mv_track_s,
                            prod=prod_s, broke=broke_s, ph=ph_s,
                            en_auto=en_auto_s, en_ff=en_ff_s, do_reset=False),
                params=st.session_state.params
            )
            st.success(f"MV_CMD={out['MV_CMD']:.3f} | MV_FF={out['MV_FF']:.3f} | MV_FB={out['MV_FB']:.3f}")
            st.write({
                "e_now": out["ERR_NOW"],
                "e_pred": out["ERR_PRED"],
                "e_eff": out["e_eff"],
                "PV_pred": out["PV_PRED"],
                "Kp": out["KP_EFF"],
                "Ti": out["TI_EFF"],
                "pTerm": out["pTerm"],
                "iTerm": out["iTerm"],
                "PV_accepted": out["PV_ACCEPTED"],
                "DT_real": out["DT_REAL_MIN"],
            })

st.caption("✅ Slope: N endpoints = N últimas médias de minuto (PV_1min). Ex.: N=5 usa as 5 últimas PV_1min.")
