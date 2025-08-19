# Resolvedor de caminho para JSONs de configuração
def _candidate_config_dirs():
    dirs = []
    try:
        dirs.append(os.getcwd())
    except Exception:
        pass
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        dirs += [base, os.path.join(base, "config"), os.path.join(base, "data")]
        parent = os.path.dirname(base)
        dirs += [parent, os.path.join(parent, "config"), os.path.join(parent, "data")]
    except Exception:
        pass
    envdir = os.environ.get("ROV_CONFIG_DIR")
    if envdir:
        dirs.insert(0, envdir)
    try:
        home = os.path.expanduser("~")
        dirs += [os.path.join(home, ".dashboard_rov")]
    except Exception:
        pass
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen and os.path.isdir(d):
            seen.add(d); out.append(d)
    return out or [os.getcwd()]

def resolve_config_path(basename: str):
    if os.path.isabs(basename) and os.path.exists(basename):
        return basename
    for d in _candidate_config_dirs():
        p = os.path.join(d, basename)
        if os.path.exists(p):
            return p
    return os.path.join(os.getcwd(), basename)

# JSON "relaxado": aceita comentários, remove vírgulas finais e BOM
def _json_relax(txt: str):
    import re, json
    if txt and txt[:1] == "\ufeff":
        txt = txt[1:]
    txt = re.sub(r'//.*?$', '', txt, flags=re.MULTILINE)      # // comments
    txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.DOTALL)    # /* ... */ comments
    txt = re.sub(r',\s*(\]|})', r'\1', txt)                # trailing commas
    return json.loads(txt)

# -*- coding: utf-8 -*-
# ============================================================
# Dashboard Operacional ROV (apenas com dados do arquivo)
# Execução: streamlit run dashboard_rov.py
# Requisitos: streamlit, pandas, plotly, numpy, python-dateutil, statsmodels (opcional)
# ============================================================

import re
import os
import json
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date
from io import BytesIO

# ---- IA (imports opcionais) ----
try:
    from sklearn.ensemble import IsolationForest
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


try:
    from prophet import Prophet  # pip install prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False


# ------------------------------
# Configuração da página
# ------------------------------
st.set_page_config(page_title="Dashboard ROV - Operação", layout="wide")

# ------------------------------
# Persistência (arquivos JSON na mesma pasta do app)
# ------------------------------
CONFIG_PATH_CATEG = os.path.join(os.getcwd(), "linhas_config.json")          # categorias (Urbana/Distrital)
CONFIG_PATH_KM    = os.path.join(os.getcwd(), "linhas_km_config.json")       # vigências de km por linha
CONFIG_PATH_VEIC  = os.path.join(os.getcwd(), "linhas_veic_config.json")     # vigências de veículos por linha

def load_json_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return _json_relax(f.read())
        except Exception:
            return {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
def save_json_config(path: str, data) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# ------------------------------
# Helpers de formatação PT-BR
# ------------------------------
def fmt_int(x):
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_float(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"
        
def trend_delta(atual, anterior, nd=1):
    if anterior is None or anterior == 0:
        return ""
    abs_delta = atual - anterior
    pct_delta = (abs_delta / anterior) * 100
    sinal = "+" if abs_delta > 0 else ""
    return f"{sinal}{fmt_int(abs_delta)} ({sinal}{fmt_float(pct_delta, nd)}%)"
    
def fmt_currency(x, nd=2):
    return f"R$ {fmt_float(x, nd)}"

def fmt_pct(x, nd=1):
    try:
        return (f"{float(x)*100:,.{nd}f}%").replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,0%"

def df_fmt_milhar(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(fmt_int)
    return df2

def df_fmt_currency(df: pd.DataFrame, cols: list[str], nd=2) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda v: fmt_currency(v, nd))
    return df2


# ------------------------------
# Utilitários de UI/Detalhes
# ------------------------------
def select_motorista_widget(candidates, key, label="Selecionar motorista"):
    """Widget padrão para selecionar um motorista dentre candidatos (lista ordenada)."""
    if not candidates:
        return None
    return st.selectbox(label, options=["(selecione)"] + candidates, index=0, key=key)

def show_motorista_details(motorista_id: str, df_scope: pd.DataFrame):
    """Mostra uma visão detalhada e visual do motorista selecionado no escopo filtrado atual."""
    if not motorista_id or motorista_id == "(selecione)":
        return

    st.markdown(f"### 🔎 Detalhes do motorista: **{motorista_id}**")

    # Filtra dados
    dfm = df_scope.copy()
    mot_col = "Cobrador/Operador" if "Cobrador/Operador" in dfm.columns else ("Matricula" if "Matricula" in dfm.columns else None)
    if mot_col is None:
        st.warning("Colunas de motorista não encontradas.")
        return
    dfm = dfm[dfm[mot_col].astype(str) == str(motorista_id)]

    if dfm.empty:
        st.info("Sem dados para o motorista no filtro atual.")
        return

    # Escolhe coluna de distância
    dist_col = "Distancia_cfg_km" if ("Distancia_cfg_km" in dfm.columns and dfm["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in dfm.columns else None)

    # KPIs
    tot_viag = len(dfm)
    tot_pax  = dfm["Passageiros"].sum() if "Passageiros" in dfm.columns else 0
    tot_km   = dfm[dist_col].sum(min_count=1) if dist_col else 0.0

    paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    present_paying = [c for c in paying_cols_all if c in dfm.columns]
    tot_pag = float(dfm[present_paying].sum().sum()) if present_paying else 0.0
    tot_grat = float(dfm["Quant Gratuidade"].sum()) if "Quant Gratuidade" in dfm.columns else 0.0

    rec_tar = tot_pag * float(tarifa_usuario)
    rec_sub = tot_pag * float(subsidio_pagante)
    rec_tot = rec_tar + rec_sub

    ipk_tot = (tot_pax / tot_km) if tot_km else 0.0
    ipk_pag = (tot_pag / tot_km) if tot_km else 0.0
    rec_km  = (rec_tot / tot_km) if tot_km else 0.0

    # Aproveitamento deste motorista
    if "Data Hora Inicio Operacao" in dfm.columns and "Data Hora Final Operacao" in dfm.columns:
        dfi = pd.to_datetime(dfm["Data Hora Inicio Operacao"], errors="coerce")
        dff = pd.to_datetime(dfm["Data Hora Final Operacao"], errors="coerce")
        dur_h = (dff - dfi).dt.total_seconds() / 3600.0
        dur_h = dur_h.where((dur_h > 0) & np.isfinite(dur_h))
        # dia de referência
        if "Data Coleta" in dfm.columns:
            dia_ref = pd.to_datetime(dfm["Data Coleta"], errors="coerce").dt.date
        else:
            dia_ref = dfi.dt.date
        util_grp = pd.DataFrame({"dia": dia_ref, "h": dur_h}).dropna().groupby("dia")["h"].sum()
        REF_HOURS = 7 + 20/60
        aproveitamento = float(util_grp.sum() / (len(util_grp) * REF_HOURS)) if len(util_grp) else 0.0
    else:
        aproveitamento = 0.0

    # Exibe KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Viagens", fmt_int(tot_viag))
    c2.metric("Passageiros", fmt_int(tot_pax))
    c3.metric("KM", fmt_float(tot_km, 1))
    c4.metric("Pagantes", fmt_int(tot_pag))
    c5.metric("Receita total", fmt_currency(rec_tot, 2))
    c6.metric("Aproveitamento", fmt_pct(aproveitamento, 1))

    # Gráficos
    g1, g2 = st.columns(2)

    # Série por dia (passageiros)
    if "Data" in dfm.columns and dfm["Data"].notna().any() and "Passageiros" in dfm.columns:
        serie = dfm.groupby("Data", as_index=False, observed=False)["Passageiros"].sum().sort_values("Data")
        fig = px.line(serie, x="Data", y="Passageiros", markers=True, title="Passageiros por dia (motorista)")
        fig.update_xaxes(tickformat="%d/%m/%Y")
        fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=320)
        g1.plotly_chart(fig, use_container_width=True)
    else:
        g1.info("Sem base de datas para série por dia.")

    # Distribuição por linha (passageiros)
    if {"Nome Linha","Passageiros"}.issubset(dfm.columns):
        by_line = dfm.groupby("Nome Linha", as_index=False, observed=False)["Passageiros"].sum().sort_values("Passageiros", ascending=False).head(12)
        fig2 = px.bar(by_line, x="Nome Linha", y="Passageiros", title="Passageiros por linha (motorista)")
        fig2.update_layout(xaxis_tickangle=-30, margin=dict(l=10,r=10,t=35,b=10), height=320)
        g2.plotly_chart(fig2, use_container_width=True)
    else:
        g2.info("Sem coluna 'Nome Linha' para a distribuição por linha.")

    # Dispersão hora x passageiros (para entender picos)
    if "Hora_Base" in dfm.columns and "Passageiros" in dfm.columns and dfm["Hora_Base"].notna().any():
        fig3 = px.scatter(dfm, x="Hora_Base", y="Passageiros", title="Demanda por hora (motorista)",
                          hover_data=[c for c in ["Nome Linha","Descricao Terminal","Data Coleta"] if c in dfm.columns])
        fig3.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)

    # Tabela detalhada (amostra)
    cols_show = [c for c in ["Data Coleta","Nome Linha","Numero Veiculo","Descricao Terminal","Passageiros", dist_col, "Quant Gratuidade"] if c and c in dfm.columns]
    if cols_show:
        st.caption("Amostra de viagens do motorista")
        df_show = dfm[cols_show].head(200).copy()
        # Formatações PT-BR
        if "Passageiros" in df_show.columns:
            df_show["Passageiros"] = df_show["Passageiros"].apply(fmt_int)
        if dist_col in df_show.columns:
            df_show[dist_col] = df_show[dist_col].apply(lambda v: fmt_float(v, 1))
        if "Quant Gratuidade" in df_show.columns:
            df_show["Quant Gratuidade"] = df_show["Quant Gratuidade"].apply(fmt_int)
        st.dataframe(df_show, use_container_width=True)


# ------------------------------
# Carregamento de dados
# ------------------------------
@st.cache_data(show_spinner=False)
def load_data(csv: object) -> pd.DataFrame:
    """Carrega o CSV (sep=';'), normaliza tipos e deriva colunas úteis.

    Aceita tanto um caminho de arquivo (str) quanto um objeto semelhante a arquivo
    retornado pelo st.file_uploader. Tenta diferentes codificações para abrir o
    arquivo e lança um erro se nenhuma delas for válida.
    """
    encodings = ["utf-8", "latin-1"]
    last_err = None
    df: Optional[pd.DataFrame] = None
    for enc in encodings:
        try:
            # Se for um objeto de arquivo, reposiciona o cursor no início antes de ler
            if hasattr(csv, "read"):
                try:
                    csv.seek(0)
                except Exception:
                    pass
                df = pd.read_csv(csv, sep=";", encoding=enc)
            else:
                df = pd.read_csv(csv, sep=";", encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise RuntimeError(f"Falha ao ler o CSV. Último erro: {last_err}")

    # limpa espaços
    df.columns = [c.strip() for c in df.columns]

    # datas
    date_cols = [
        "Data Coleta",
        "Data Hora Inicio Operacao",
        "Data Hora Final Operacao",
        "Data Hora Saida Terminal",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # numéricos
    numeric_candidates = [
        "Passageiros",
        "Distancia",
        "Num Terminal Viagem",
        "Catraca Inicial",
        "Catraca Final",
        "Catraca Inicial.1",
        "Catraca Final.1",
        "Total Fichas",
        "Catraca Pendente",
        "Ordem",
        # Tarifas
        "Quant Gratuidade",
        "Quant Passagem",
        "Quant Passagem Integracao",
        "Quant Passe",
        "Quant Passe Integracao",
        "Quant Vale Transporte",
        "Quant Vale Transporte Integracao",
        "Quant Inteiras",
    ]
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # categóricos comuns
    cat_candidates = [
        "Nome Linha",
        "Codigo Externo Linha",
        "Codigo Interno Linha",
        "Numero Veiculo",
        "Codigo Veiculo",
        "Nome Garagem",
        "Descricao Terminal",
        "Periodo Operacao",
        "Grupo Veiculo",
        "Descricao Tipo Evento",
        "Tipo Viagem",
        "Viagem",
        "Sub Linha",
        "Cobrador/Operador",
        "Nome Operadora",
        "Orgao Gestor",
        "Codigo Operadora",
        "Matricula"
    ]
    for c in cat_candidates:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # feature engineering leve
    if "Data Coleta" in df.columns:
        df["Data"] = df["Data Coleta"].dt.date
        df["Ano"] = df["Data Coleta"].dt.year
        df["Mes"] = df["Data Coleta"].dt.month
        df["Dia"] = df["Data Coleta"].dt.day
        try:
            df["DiaSemana"] = df["Data Coleta"].dt.day_name(locale="pt_BR")
        except Exception:
            df["DiaSemana"] = df["Data Coleta"].dt.day_of_week
        df["Hora"] = df["Data Coleta"].dt.hour
    else:
        df["Data"] = pd.NaT

    # base para heatmap (robusto)
    candidates = [
        "Data Hora Saida Terminal",
        "Data Hora Inicio Operacao",
        "Data Coleta",
    ]
    time_basis = None
    max_nonnull = -1
    for c in candidates:
        if c in df.columns:
            if not np.issubdtype(df[c].dtype, np.datetime64):
                df[c] = pd.to_datetime(df[c], errors="coerce")
            nonnull_count = df[c].notna().sum()
            if nonnull_count > max_nonnull:
                max_nonnull = nonnull_count
                time_basis = c
    if time_basis is not None and max_nonnull > 0:
        df["Hora_Base"] = df[time_basis].dt.hour
        df["DiaSemana_Base"] = df[time_basis].dt.dayofweek
    else:
        if "Hora" in df.columns and df["Hora"].notna().any():
            df["Hora_Base"] = pd.to_numeric(df["Hora"], errors="coerce")
            if "Data Coleta" in df.columns:
                if not np.issubdtype(df["Data Coleta"].dtype, np.datetime64):
                    df["Data Coleta"] = pd.to_datetime(df["Data Coleta"], errors="coerce")
                df["DiaSemana_Base"] = df["Data Coleta"].dt.dayofweek
            else:
                df["DiaSemana_Base"] = np.nan
        else:
            df["Hora_Base"] = np.nan
            df["DiaSemana_Base"] = np.nan

    return df

# ------------------------------
# KM por linha com vigências
# ------------------------------
BASE_DATE_FOR_FALLBACK = pd.Timestamp("2025-07-08")

def load_km_store() -> dict:
    return load_json_config(resolve_config_path("linhas_km_config.json"))

def save_km_store(store: dict) -> bool:
    return save_json_config(resolve_config_path("linhas_km_config.json"), store)

def append_km_record(store: dict, line: str, inicio: pd.Timestamp, km: float, fim: Optional[pd.Timestamp] = None):
    """Acrescenta uma vigência. Se houver período em aberto, encerra no dia anterior ao novo início."""
    line = str(line)
    vlist = store.get(line, [])
    # encerra período aberto (fim None)
    try:
        open_idx = next((i for i, r in enumerate(vlist) if r.get("fim") in (None, "")), None)
    except Exception:
        open_idx = None
    if open_idx is not None:
        try:
            end_date = (inicio - pd.Timedelta(days=1)).date()
            vlist[open_idx]["fim"] = str(end_date)
        except Exception:
            pass
    rec = {"inicio": str(pd.Timestamp(inicio).date()),
           "fim": (str(pd.Timestamp(fim).date()) if (fim is not None and pd.notna(fim)) else None),
           "km": float(km)}
    vlist.append(rec)
    # ordena
    vlist.sort(key=lambda r: r.get("inicio") or "9999-12-31")
    store[line] = vlist

def km_for_date(vlist: list, when: pd.Timestamp) -> Optional[float]:
    """Retorna km vigente na data 'when'; se não houver, tenta a vigência válida em 08/07/2025."""
    when = pd.to_datetime(when, errors="coerce")
    if vlist and pd.notna(when):
        for rec in vlist:
            ini = pd.to_datetime(rec.get("inicio"), errors="coerce")
            fim = pd.to_datetime(rec.get("fim"), errors="coerce") if rec.get("fim") else pd.NaT
            if pd.notna(ini) and (when >= ini) and (pd.isna(fim) or when <= fim):
                return float(rec.get("km"))
        # fallback base date
        for rec in vlist:
            ini = pd.to_datetime(rec.get("inicio"), errors="coerce")
            fim = pd.to_datetime(rec.get("fim"), errors="coerce") if rec.get("fim") else pd.NaT
            if pd.notna(ini) and (BASE_DATE_FOR_FALLBACK >= ini) and (pd.isna(fim) or BASE_DATE_FOR_FALLBACK <= fim):
                return float(rec.get("km"))
    return None

def apply_km_vigente(df_in: pd.DataFrame, store: dict) -> pd.DataFrame:
    if "Nome Linha" not in df_in.columns:
        df_in["Distancia_cfg_km"] = np.nan
        return df_in
    # data de referência
    ref_col = None
    for c in ["Data Coleta", "Data", "Data Hora Saida Terminal", "Data Hora Inicio Operacao"]:
        if c in df_in.columns:
            ref_col = c
            break
    if ref_col is None:
        df_in["Distancia_cfg_km"] = np.nan
        return df_in
    def _row_km(row):
        vlist = store.get(str(row["Nome Linha"]), [])
        val = km_for_date(vlist, row[ref_col])
        return np.nan if val is None else val
    df_in["Distancia_cfg_km"] = df_in.apply(_row_km, axis=1)
    return df_in

# ------------------------------
# Veículos por linha com vigências (configurados/em operação)
# ------------------------------
def load_veic_store() -> dict:
    return load_json_config(resolve_config_path("linhas_veic_config.json"))

def save_veic_store(store: dict) -> bool:
    return save_json_config(resolve_config_path("linhas_veic_config.json"), store)

def append_veic_record(store: dict, line: str, inicio: pd.Timestamp, veic: float, fim: Optional[pd.Timestamp] = None):
    line = str(line)
    vlist = store.get(line, [])
    # encerra período aberto
    try:
        open_idx = next((i for i, r in enumerate(vlist) if r.get("fim") in (None, "")), None)
    except Exception:
        open_idx = None
    if open_idx is not None:
        try:
            end_date = (inicio - pd.Timedelta(days=1)).date()
            vlist[open_idx]["fim"] = str(end_date)
        except Exception:
            pass
    rec = {"inicio": str(pd.Timestamp(inicio).date()),
           "fim": (str(pd.Timestamp(fim).date()) if (fim is not None and pd.notna(fim)) else None),
           "veiculos": float(veic)}
    vlist.append(rec)
    vlist.sort(key=lambda r: r.get("inicio") or "9999-12-31")
    store[line] = vlist

def veic_for_date(vlist: list, when: pd.Timestamp) -> Optional[float]:
    when = pd.to_datetime(when, errors="coerce")
    if vlist and pd.notna(when):
        for rec in vlist:
            ini = pd.to_datetime(rec.get("inicio"), errors="coerce")
            fim = pd.to_datetime(rec.get("fim"), errors="coerce") if rec.get("fim") else pd.NaT
            if pd.notna(ini) and (when >= ini) and (pd.isna(fim) or when <= fim):
                return float(rec.get("veiculos"))
        # fallback mesma base do KM
        for rec in vlist:
            ini = pd.to_datetime(rec.get("inicio"), errors="coerce")
            fim = pd.to_datetime(rec.get("fim"), errors="coerce") if rec.get("fim") else pd.NaT
            if pd.notna(ini) and (BASE_DATE_FOR_FALLBACK >= ini) and (pd.isna(fim) or BASE_DATE_FOR_FALLBACK <= fim):
                return float(rec.get("veiculos"))
    return None

def apply_veic_vigente(df_in: pd.DataFrame, store: dict) -> pd.DataFrame:
    if "Nome Linha" not in df_in.columns:
        df_in["Veiculos_cfg"] = np.nan
        return df_in
    ref_col = None
    for c in ["Data Coleta", "Data", "Data Hora Saida Terminal", "Data Hora Inicio Operacao"]:
        if c in df_in.columns:
            ref_col = c
            break
    if ref_col is None:
        df_in["Veiculos_cfg"] = np.nan
        return df_in
    def _row_v(row):
        vlist = store.get(str(row["Nome Linha"]), [])
        val = veic_for_date(vlist, row[ref_col])
        return np.nan if val is None else val
    df_in["Veiculos_cfg"] = df_in.apply(_row_v, axis=1)
    return df_in

# ------------------------------
# Entrada do arquivo
# ------------------------------
# Override opcional via upload de JSONs
with st.sidebar.expander("🔧 Parâmetros (JSON)", expanded=False):
    cfg_upload_categ = st.file_uploader("linhas_config.json", type=["json"], key="upload_categ")
    cfg_upload_km    = st.file_uploader("linhas_km_config.json", type=["json"], key="upload_km")
    cfg_upload_veic  = st.file_uploader("linhas_veic_config.json", type=["json"], key="upload_veic")
    if cfg_upload_categ is not None:
        try:
            cfg_categ = _json_relax(cfg_upload_categ.read().decode("utf-8", "ignore"))
        except Exception as e:
            st.warning(f"Falha ao ler linhas_config.json enviado: {e}")
    if cfg_upload_km is not None:
        try:
            km_store = _json_relax(cfg_upload_km.read().decode("utf-8", "ignore"))
        except Exception as e:
            st.warning(f"Falha ao ler linhas_km_config.json enviado: {e}")
    if cfg_upload_veic is not None:
        try:
            veic_store = _json_relax(cfg_upload_veic.read().decode("utf-8", "ignore"))
        except Exception as e:
            st.warning(f"Falha ao ler linhas_veic_config.json enviado: {e}")

st.sidebar.title("⚙️ Configurações")
# Campo para upload de arquivo CSV pelo usuário
uploaded_file = st.sidebar.file_uploader("Carregue o arquivo de dados (CSV ';')", type=["csv"])
if uploaded_file is None:
    st.sidebar.info("Por favor, faça upload do arquivo CSV.")
    st.stop()

with st.spinner("Carregando dados..."):
    df = load_data(uploaded_file)

st.title("📊 Dashboard Operacional ROV")
st.caption("*Baseado exclusivamente nos dados existentes do arquivo .CSV importado*")

# ------------------------------
# Classificação de Linhas (Urbana/Distrital) com persistência
# ------------------------------
if "Nome Linha" in df.columns:
    st.sidebar.header("Classificação de Linhas")
    cfg_categ = load_json_config(resolve_config_path("linhas_config.json"))
    valid_vals = {"Urbana", "Distrital"}
    cfg_categ = {k: (v if v in valid_vals else None) for k, v in cfg_categ.items() if isinstance(k, str)}
    linhas_unicas = sorted([x for x in df["Nome Linha"].dropna().astype(str).unique()])

    df["Categoria Linha"] = df["Nome Linha"].map(cfg_categ)
    atuais_urbanas = [l for l in linhas_unicas if cfg_categ.get(l) == "Urbana"]
    atuais_distritais = [l for l in linhas_unicas if cfg_categ.get(l) == "Distrital"]

    with st.sidebar.expander("Apropriar Linhas por Categoria", expanded=False):
        sel_urbanas = st.multiselect("Linhas Urbanas", options=linhas_unicas, default=atuais_urbanas, key="ms_urb")
        restantes = [l for l in linhas_unicas if l not in sel_urbanas]
        sel_distritais = st.multiselect("Linhas Distritais", options=restantes, default=[l for l in atuais_distritais if l in restantes], key="ms_dis")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("💾 Salvar classificação", use_container_width=True):
                novo_cfg = {l: "Urbana" for l in sel_urbanas}
                for l in sel_distritais:
                    if l not in novo_cfg:
                        novo_cfg[l] = "Distrital"
                if save_json_config(resolve_config_path("linhas_config.json"), novo_cfg):
                    st.success("Classificação salva.")
                    cfg_categ = novo_cfg
        with col_s2:
            if st.button("↩️ Reset (limpar)", use_container_width=True):
                if save_json_config(resolve_config_path("linhas_config.json"), {}):
                    st.warning("Classificação removida.")
                    cfg_categ = {}

    df["Categoria Linha"] = df["Nome Linha"].map(cfg_categ)

# ------------------------------
# KM por Linha (vigências) com persistência
# ------------------------------
km_store = load_km_store()
if "Nome Linha" in df.columns:
    with st.sidebar.expander("KM por Linha (vigências)", expanded=False):
        linhas_unicas = sorted([x for x in df["Nome Linha"].dropna().astype(str).unique()])
        linha_sel = st.selectbox("Linha", options=["(selecione)"] + linhas_unicas, index=0, key="sel_linha_km")
        if linha_sel and linha_sel != "(selecione)":
            vlist = km_store.get(linha_sel, [])
            if vlist:
                st.write("Vigências atuais:")
                st.dataframe(pd.DataFrame(vlist))
            else:
                st.info("Sem vigências cadastradas para esta linha.")

            st.markdown("**Adicionar/alterar vigência**")
            colk1, colk2 = st.columns(2)
            with colk1:
                inicio_input = st.date_input("Início da vigência", value=date.today(), key="km_inicio")
                km_input = st.number_input("KM da linha", min_value=0.0, value=0.0, step=0.1, format="%.2f", key="km_val")
            with colk2:
                fim_enable = st.checkbox("Definir fim da vigência?", value=False, key="km_fim_chk")
                fim_input = st.date_input("Fim da vigência", value=date.today(), key="km_fim") if fim_enable else None

            c1, c2 = st.columns(2)
            with c1:
                if st.button("➕ Registrar vigência", use_container_width=True):
                    try:
                        inicio_ts = pd.Timestamp(inicio_input)
                        fim_ts = pd.Timestamp(fim_input) if fim_enable and fim_input else pd.NaT
                        append_km_record(km_store, linha_sel, inicio_ts, km_input, fim=fim_ts)
                        if save_km_store(km_store):
                            st.success("Vigência registrada/salva.")
                        else:
                            st.error("Falha ao salvar vigência.")
                    except Exception as e:
                        st.error(f"Erro ao registrar: {e}")
            with c2:
                if st.button("🗑️ Limpar todas vigências da linha", use_container_width=True):
                    km_store[linha_sel] = []
                    if save_km_store(km_store):
                        st.warning("Vigências removidas para a linha.")
                    else:
                        st.error("Falha ao salvar após limpeza.")

# aplica coluna de distância configurada (por linha+data)
df = apply_km_vigente(df, km_store)

# ------------------------------
# Veículos por Linha (vigências) com persistência
# ------------------------------
veic_store = load_veic_store()
if "Nome Linha" in df.columns:
    with st.sidebar.expander("Veículos por Linha (vigências)", expanded=False):
        linhas_unicas = sorted([x for x in df["Nome Linha"].dropna().astype(str).unique()])
        linha_sel_v = st.selectbox("Linha", options=["(selecione)"] + linhas_unicas, index=0, key="sel_linha_veic")
        if linha_sel_v and linha_sel_v != "(selecione)":
            vlist_v = veic_store.get(linha_sel_v, [])
            if vlist_v:
                st.write("Vigências atuais (veículos):")
                st.dataframe(pd.DataFrame(vlist_v))
            else:
                st.info("Sem vigências cadastradas para esta linha (veículos).")

            st.markdown("**Adicionar/alterar vigência (veículos)**")
            colv1, colv2 = st.columns(2)
            with colv1:
                inicio_v = st.date_input("Início da vigência", value=date.today(), key="veic_inicio")
                veic_qtd = st.number_input("Veículos em operação", min_value=0.0, value=0.0, step=1.0, format="%.0f", key="veic_val")
            with colv2:
                fim_v_enable = st.checkbox("Definir fim da vigência?", value=False, key="veic_fim_chk")
                fim_v = st.date_input("Fim da vigência", value=date.today(), key="veic_fim") if fim_v_enable else None

            cv1, cv2 = st.columns(2)
            with cv1:
                if st.button("➕ Registrar vigência (veículos)", use_container_width=True):
                    try:
                        inicio_ts = pd.Timestamp(inicio_v)
                        fim_ts = pd.Timestamp(fim_v) if fim_v_enable and fim_v else pd.NaT
                        append_veic_record(veic_store, linha_sel_v, inicio_ts, veic_qtd, fim=fim_ts)
                        if save_veic_store(veic_store):
                            st.success("Vigência de veículos registrada/salva.")
                        else:
                            st.error("Falha ao salvar vigência de veículos.")
                    except Exception as e:
                        st.error(f"Erro ao registrar veículos: {e}")
            with cv2:
                if st.button("🗑️ Limpar vigências (veículos) da linha", use_container_width=True):
                    veic_store[linha_sel_v] = []
                    if save_veic_store(veic_store):
                        st.warning("Vigências de veículos removidas para a linha.")
                    else:
                        st.error("Falha ao salvar após limpeza (veículos).")

# aplica coluna de veículos configurados
df = apply_veic_vigente(df, veic_store)

# ------------------------------
# Filtros
# ------------------------------
st.sidebar.header("Filtros")
df_filtered = df.copy()

# Período
if "Data Coleta" in df_filtered.columns and df_filtered["Data Coleta"].notna().any():
    min_d = pd.to_datetime(df_filtered["Data Coleta"].min()).date()
    max_d = pd.to_datetime(df_filtered["Data Coleta"].max()).date()
    d_ini, d_fim = st.sidebar.date_input("Período", value=(min_d, max_d), min_value=min_d, max_value=max_d, format="DD/MM/YYYY")
    if isinstance(d_ini, date) and isinstance(d_fim, date):
        mask = (df_filtered["Data Coleta"].dt.date >= d_ini) & (df_filtered["Data Coleta"].dt.date <= d_fim)
        df_filtered = df_filtered.loc[mask]

# Linha
if "Nome Linha" in df_filtered.columns:
    linhas = sorted([x for x in df_filtered["Nome Linha"].dropna().unique().tolist()])
    sel_linhas = st.sidebar.multiselect("Linhas", linhas)
    if sel_linhas:
        df_filtered = df_filtered[df_filtered["Nome Linha"].isin(sel_linhas)]

# Categoria

if "Categoria Linha" in df_filtered.columns:
    import unicodedata
    def _norm_txt(s):
        try: s = str(s)
        except: s = ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.strip().lower()
    _aliases = {"urbanas":"urbana","urbana":"urbana","urbano":"urbana",
                "distritais":"distrital","distrital":"distrital","distrito":"distrital"}
    cat_series = df_filtered["Categoria Linha"].astype(str)
    present_keys = {}
    for v in cat_series.dropna().unique():
        k = _aliases.get(_norm_txt(v), _norm_txt(v))
        present_keys.setdefault(k, v)
    display_opts = ["Todas"] + sorted(present_keys.values())
    cat_opt = st.sidebar.selectbox("Categoria", options=display_opts, index=0)
    if cat_opt != "Todas":
        sel_key = _aliases.get(_norm_txt(cat_opt), _norm_txt(cat_opt))
        mask = cat_series.map(lambda x: _aliases.get(_norm_txt(x), _norm_txt(x))) == sel_key
        if mask.any():
            df_filtered = df_filtered[mask]
        else:
            st.warning("Nenhum registro encontrado para a categoria selecionada. Mantendo todos os registros.")







# === PAINEL DE SUSPEITAS: GRATUIDADES POR MOTORISTA ===
import pandas as _pd
import numpy as _np
import re as _re
try:
    import streamlit as _st
    import plotly.graph_objects as _go
except Exception:
    _st = None

def _sus_first(df, names):
    for n in names:
        if isinstance(n, str) and n in df.columns:
            return n
    return None

def _force_str(col):
    if isinstance(col, list) or isinstance(col, tuple):
        return col[0] if col else None
    return col

def _sus_detect_cols(df):
    # Preferir NOME do motorista
    name_candidates = ["Cobrador/Operador","Nome Motorista","Motorista","Nome do Motorista","Nome Condutor","Condutor"]
    id_candidates   = ["Matricula","Matrícula","CPF Motorista","ID Motorista","Id Motorista"]
    driver_name = _sus_first(df, name_candidates)
    driver_id   = _sus_first(df, id_candidates)
    driver_col  = driver_name or driver_id  # string esperada
    driver_col  = _force_str(driver_col)
    driver_name = _force_str(driver_name)
    driver_id   = _force_str(driver_id)

    line_col    = _sus_first(df, ["Nome Linha","Linha"])
    line_col    = _force_str(line_col)

    # Pagantes SEM integração
    pay_whitelist = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    pay_present = [c for c in pay_whitelist if c in df.columns]
    if not pay_present:
        for c in df.columns:
            if _pd.api.types.is_numeric_dtype(df[c]):
                if _re.search(r"(?i)quant", c) and _re.search(r"(?i)(inteir|passag|passe|vale|vt)", c) and not _re.search(r"(?i)grat", c) and not _re.search(r"(?i)integr", c):
                    pay_present.append(c)

    # Gratuidades SEM integração
    grat_cols = [c for c in df.columns if _re.search(r"(?i)grat", c) and not _re.search(r"(?i)integr", c)]

    return driver_col, driver_name, driver_id, line_col, pay_present, grat_cols

def _robust_baseline(series):
    s = _pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return _np.nan, _np.nan
    med = s.median()
    mad = (s - med).abs().median()
    madn = 1.4826 * mad if (mad is not None and mad > 0) else _np.nan
    return med, madn

def _sus_compute_table(df, min_trips=10, min_pag=100, baseline="linha"):
    if df is None or df.empty:
        return _pd.DataFrame(), "Sem dados após filtros."

    drv, drv_name, drv_id, lin, pay_cols, grat_cols = _sus_detect_cols(df)

    if not isinstance(drv, str) or drv not in df.columns:
        return _pd.DataFrame(), "Coluna de motorista (nome/matrícula) não encontrada."
    if not pay_cols:
        return _pd.DataFrame(), "Colunas de pagantes (sem integração) não localizadas."
    if not grat_cols:
        return _pd.DataFrame(), "Colunas de gratuidade (sem integração) não localizadas."

    d = df.copy()
    d["_pag"] = _pd.to_numeric(d[pay_cols].sum(axis=1), errors="coerce").fillna(0)
    d["_grat"] = _pd.to_numeric(d[grat_cols].sum(axis=1), errors="coerce").fillna(0)
    d["_one"] = 1

    grp_keys = [k for k in [drv, lin] if isinstance(k, str)]
    if not grp_keys:
        grp_keys = [drv]

    agg = d.groupby(grp_keys, dropna=False).agg(
        viagens=("_one","sum"),
        pagantes=("_pag","sum"),
        gratuidades=("_grat","sum"),
    ).reset_index()

    agg["grat_pag_ratio"] = _np.where(agg["pagantes"]>0, agg["gratuidades"]/agg["pagantes"], _np.nan)
    if "grat_pag_ratio" not in agg.columns:
        agg["grat_pag_ratio"] = _np.nan

    agg = agg[(agg["viagens"] >= int(min_trips)) & (agg["pagantes"] >= float(min_pag))]
    if agg.empty:
        return agg, "Sem grupos com amostragem mínima (ajuste os limiares)."

    if lin and isinstance(lin, str) and lin in agg.columns:
        med = agg.groupby(lin)["grat_pag_ratio"].transform(lambda s: _pd.to_numeric(s, errors="coerce").median())
        def _madn_tr(s):
            s2 = _pd.to_numeric(s, errors="coerce").dropna()
            if s2.empty: return _np.nan
            m = s2.median()
            return 1.4826 * (s2 - m).abs().median() if _pd.notna(m) else _np.nan
        madn = agg.groupby(lin)["grat_pag_ratio"].transform(_madn_tr)
    else:
        gsr = _pd.to_numeric(agg["grat_pag_ratio"], errors="coerce")
        med_val, madn_val = _robust_baseline(gsr)
        med = _pd.Series(med_val, index=agg.index)
        madn = _pd.Series(madn_val, index=agg.index)

    denom = madn.replace({0:_np.nan})
    agg["z_rob"] = ( _pd.to_numeric(agg["grat_pag_ratio"], errors="coerce") - _pd.to_numeric(med, errors="coerce") ) / denom

    def _lvl(z):
        if _pd.isna(z): return "inconclusivo"
        if z >= 3: return "ALTA"
        if z >= 2: return "MÉDIA"
        return "BAIXA"
    def _badge(s):
        return {"ALTA":"🔴 ALTA", "MÉDIA":"🟠 MÉDIA", "BAIXA":"🟡 BAIXA", "inconclusivo":"⚪ Inconclusivo"}.get(s, "⚪ Inconclusivo")

    agg["Suspeita"] = agg["z_rob"].apply(_lvl)
    agg["Sinal"] = agg["Suspeita"].apply(_badge)
    agg["% grat/pag"] = (_pd.to_numeric(agg["grat_pag_ratio"], errors="coerce")*100).round(2)

    order = _pd.CategoricalDtype(categories=["ALTA","MÉDIA","BAIXA","inconclusivo"], ordered=True)
    agg["Suspeita"] = agg["Suspeita"].astype(order)
    agg = agg.sort_values(["Suspeita","z_rob","% grat/pag"], ascending=[True, False, False])

    # Seleção – garante "Motorista" por NOME quando existir
    out_cols = [c for c in [drv, lin, "viagens","pagantes","gratuidades","% grat/pag","z_rob","Sinal"] if isinstance(c, str) or c in ["viagens","pagantes","gratuidades","% grat/pag","z_rob","Sinal"]]
    out = agg[out_cols].copy()
    if drv_name and isinstance(drv_name, str) and drv_name in out.columns:
        out.rename(columns={drv_name:"Motorista"}, inplace=True)
    elif drv in out.columns:
        out.rename(columns={drv:"Motorista"}, inplace=True)
    if lin and isinstance(lin, str) and lin in out.columns:
        out.rename(columns={lin:"Linha"}, inplace=True)
    return out, None

def _sus_trip_level(df, selected_driver):
    drv, drv_name, drv_id, lin, pay_cols, grat_cols = _sus_detect_cols(df)
    if not isinstance(drv, str) or drv not in df.columns or not pay_cols or not grat_cols:
        return _pd.DataFrame(), "Colunas necessárias não localizadas."

    d = df.copy()
    date_col = _sus_first(d, ["Data","Data Coleta","DataColeta"]) or "Data"
    if date_col in d.columns:
        d[date_col] = _pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)

    ini_col = _sus_first(d, ["Data Hora Inicio Operacao","Data Hora Início Operação","Inicio Operacao","Início Operação","Hora Inicio","DataHoraInicio"])
    fim_col = _sus_first(d, ["Data Hora Final Operacao","Data Hora Final Operação","Fim Operacao","Hora Final","DataHoraFim"])
    if ini_col in d.columns: d[ini_col] = _pd.to_datetime(d[ini_col], errors="coerce", dayfirst=True)
    if fim_col in d.columns: d[fim_col] = _pd.to_datetime(d[fim_col], errors="coerce", dayfirst=True)

    d = d[d[drv] == selected_driver].copy()
    if d.empty:
        return _pd.DataFrame(), "Sem viagens para o motorista selecionado."

    d["_pag"] = _pd.to_numeric(d[pay_cols].sum(axis=1), errors="coerce").fillna(0)
    d["_grat"] = _pd.to_numeric(d[grat_cols].sum(axis=1), errors="coerce").fillna(0)
    d["grat_pag_ratio"] = _np.where(d["_pag"]>0, d["_grat"]/d["_pag"], _np.nan)

    if lin and isinstance(lin, str) and lin in d.columns:
        med = d.groupby(lin)["grat_pag_ratio"].transform(lambda s: _pd.to_numeric(s, errors="coerce").median())
        def _madn_tr(s):
            s2 = _pd.to_numeric(s, errors="coerce").dropna()
            if s2.empty: return _np.nan
            m = s2.median()
            return 1.4826 * (s2 - m).abs().median() if _pd.notna(m) else _np.nan
        madn = d.groupby(lin)["grat_pag_ratio"].transform(_madn_tr)
    else:
        med_val, madn_val = _robust_baseline(d["grat_pag_ratio"])
        med = _pd.Series(med_val, index=d.index)
        madn = _pd.Series(madn_val, index=d.index)

    denom = madn.replace({0:_np.nan})
    d["z_rob_viagem"] = ( _pd.to_numeric(d["grat_pag_ratio"], errors="coerce") - _pd.to_numeric(med, errors="coerce") ) / denom

    def _lvl(z):
        if _pd.isna(z): return "inconclusivo"
        if z >= 3: return "ALTA"
        if z >= 2: return "MÉDIA"
        return "BAIXA"
    d["Suspeita (viagem)"] = d["z_rob_viagem"].apply(_lvl)

    if ini_col and fim_col and ini_col in d.columns and fim_col in d.columns:
        d["Duração (min)"] = (d[fim_col] - d[ini_col]).dt.total_seconds() / 60.0

    show_cols = []
    if "Data" in d.columns: show_cols.append("Data")
    elif date_col in d.columns: show_cols.append(date_col)
    if lin and isinstance(lin, str) and lin in d.columns: show_cols.append(lin)
    if ini_col in d.columns: show_cols.append(ini_col)
    if fim_col in d.columns: show_cols.append(fim_col)
    show_cols += ["_pag","_grat","grat_pag_ratio","z_rob_viagem","Suspeita (viagem)"]
    out = d[show_cols].copy()
    rename_map = {}
    if date_col in out.columns: rename_map[date_col] = "Data"
    if lin and isinstance(lin, str) and lin in out.columns: rename_map[lin] = "Linha"
    if ini_col in out.columns: rename_map[ini_col] = "Início"
    if fim_col in out.columns: rename_map[fim_col] = "Fim"
    rename_map.update({
        "_pag":"Pagantes",
        "_grat":"Gratuidades",
        "grat_pag_ratio":"% grat/pag (viagem)",
        "z_rob_viagem":"z_rob (viagem)"
    })
    out.rename(columns=rename_map, inplace=True)
    out["% grat/pag (viagem)"] = (out["% grat/pag (viagem)"]*100).round(2)
    return out.sort_values(["Suspeita (viagem)","% grat/pag (viagem)"], ascending=[True, False]), None

def _render_suspeitas_panel(df):
    if _st is None:
        return

    _st.markdown("## 🚩 Possíveis desvios: gratuidades por motorista")
    _st.caption(
        "Painel de **gratuidades (sem integrações) ÷ pagantes (sem integrações)** por **motorista**. "
        "Baseline **robusto** (mediana + MAD) por **linha**. Use os filtros gerais do dashboard para refinar o escopo."
    )

    _st.sidebar.markdown("**Parâmetros de suspeição**")
    min_trips = int(_st.sidebar.number_input("Mínimo de viagens por motorista", min_value=1, max_value=100, value=10, step=1))
    min_pag = float(_st.sidebar.number_input("Mínimo de pagantes (sem integr.)", min_value=0.0, max_value=1e6, value=100.0, step=10.0))
    baseline = _st.sidebar.selectbox("Baseline", ["Por linha (recomendado)","Global"], index=0)
    baseline_key = "linha" if baseline.startswith("Por linha") else "global"
    topn = int(_st.sidebar.number_input("Top N por suspeita", min_value=5, max_value=100, value=20, step=5))

    base_df = df.copy()
    tbl, warn = _sus_compute_table(base_df, min_trips=min_trips, min_pag=min_pag, baseline=baseline_key)
    if warn:
        _st.info(warn); return
    if tbl.empty:
        _st.info("Sem registros após aplicar os parâmetros."); return

    _st.markdown("### Ranking por motorista")
    _st.dataframe(tbl.head(topn), use_container_width=True)

    try:
        level_color = {"ALTA":"#ef4444","MÉDIA":"#f97316","BAIXA":"#facc15","inconclusivo":"#9ca3af"}
        chart = tbl.head(topn).copy()
        if "Motorista" not in chart.columns:
            # fallback: tenta detectar o campo de nome
            name_col = _sus_first(chart, ["Cobrador/Operador","Nome Motorista","Motorista","Nome do Motorista","Nome Condutor","Condutor"]) or chart.columns[0]
            chart.rename(columns={name_col:"Motorista"}, inplace=True)
        chart["Nivel"] = chart["Sinal"].astype(str).str.split().str[-1].map({"ALTA":"ALTA","MÉDIA":"MÉDIA","BAIXA":"BAIXA"}).fillna("BAIXA")
        chart["Cor"] = chart["Nivel"].map(level_color)
        x = chart["Motorista"]
        fig = _go.Figure()
        fig.add_bar(x=x, y=chart["% grat/pag"], marker_color=chart["Cor"], name="% grat/pag")
        fig.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0), yaxis_title="% grat/pag", xaxis_title="Motorista")
        _st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as _e:
        _st.warning(f"Falha ao desenhar gráfico: {_e}")

    mot_opts = tbl["Motorista"].dropna().unique().tolist() if "Motorista" in tbl.columns else []
    if mot_opts:
        sel = _st.selectbox("🔍 Detalhar motorista", mot_opts, index=0)
        if sel:
            _st.markdown(f"#### Viagens de **{sel}**")
            det, warn2 = _sus_trip_level(base_df, sel)
            if warn2:
                _st.info(warn2)
            elif det.empty:
                _st.info("Sem viagens para o motorista selecionado.")
            else:
                _st.dataframe(det, use_container_width=True)
                _st.caption("Viagens **ALTA/MÉDIA** indicam maior probabilidade de desvio; confirme o contexto operacional.")

    _st.markdown("**Legenda:** 🔴 ALTA (z ≥ 3) • 🟠 MÉDIA (2 ≤ z < 3) • 🟡 BAIXA (0 ≤ z < 2) • ⚪ Inconclusivo (amostra mínima/variância).")

try:
    _df_base = df_filtered.copy() if 'df_filtered' in globals() else df.copy()
    _render_suspeitas_panel(_df_base)
except Exception as _e:
    try:
        _st.warning(f"Falha ao renderizar painel de suspeitas: {_e}")
    except Exception:
        pass




# === Painel Rotatividade Motoristas x Veículos (injetado) ===
VEIC_CANDIDATES = [
    "Numero Veiculo"
]
MOT_CANDIDATES = [
    "Cobrador/Operador"
]
DT_CANDIDATES = [
    "Data Hora Inicio Operacao"
]

def _find_col(df, candidates):
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
    for cand in candidates:
        for col_lower, original in lower_map.items():
            if str(cand).lower() == col_lower:
                return original
    return None

def _kpi_value(v) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        try:
            return f"{float(v):,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return str(v)

def _default_index(cols, keywords):
    cols = list(cols)
    for i, c in enumerate(cols):
        name = str(c).lower()
        if any(k in name for k in keywords):
            return i
    return 0 if cols else 0

def show_rotatividade_motoristas_por_veiculo(
    df,
    veic_col=None,
    mot_col=None,
    dt_col=None,
    titulo="Rotatividade de Motoristas por Veículo"
):
    if df is None or df.empty:
        st.info("Sem dados no período selecionado.")
        return

    # 1) Tentativa de auto-detecção
    vcol = veic_col or _find_col(df, VEIC_CANDIDATES)
    mcol = mot_col or _find_col(df, MOT_CANDIDATES)
    dcol = dt_col or _find_col(df, DT_CANDIDATES)

    # 2) Se não encontrar, permitir seleção manual via UI (sem travar a execução)
    if vcol is None or mcol is None:
        st.warning("Não foi possível identificar as colunas de **veículo** e/ou **motorista**. Selecione abaixo.")
        with st.expander("Selecionar colunas manualmente", expanded=True):
            cols = list(df.columns)
            v_idx = _default_index(cols, ["veic","veículo","veiculo","carro","prefixo","placa","bus","ônibus","onibus"])
            m_idx = _default_index(cols, ["motor","operador","cobrador","matric","cpf"])
            vcol = st.selectbox("Coluna de Veículo", cols, index=min(v_idx, len(cols)-1), key="rot_pick_vcol")
            mcol = st.selectbox("Coluna de Motorista", cols, index=min(m_idx, len(cols)-1), key="rot_pick_mcol")
            # Data/hora opcional
            d_idx = _default_index(cols, ["data","hora","timestamp","dt","datetime"])
            usar_data = st.checkbox("Usar coluna de data/hora (opcional)", value=(d_idx is not None and len(cols)>0), key="rot_use_dt")
            if usar_data and len(cols)>0:
                dcol = st.selectbox("Coluna de Data/Hora", cols, index=min(d_idx, len(cols)-1), key="rot_pick_dcol")
            else:
                dcol = None

    # Se ainda assim faltou algo essencial, aborta com aviso claro
    if vcol is None or mcol is None:
        st.stop()

    work = df[[vcol, mcol]].copy()
    work[vcol] = work[vcol].astype(str).str.strip()
    work[mcol] = work[mcol].astype(str).str.strip()
    work = work[(work[vcol] != "") & (work[mcol] != "")].dropna(subset=[vcol, mcol])

    agg = (
        work.groupby(vcol, dropna=False)[mcol]
            .agg(
                qtd_motoristas="nunique",
                lista_motoristas=lambda s: sorted(set(s.astype(str)))
            )
            .reset_index()
    )
    agg["trocas_est"] = (agg["qtd_motoristas"] - 1).clip(lower=0)

    if dcol and dcol in df.columns:
        tmp = df[[vcol, mcol, dcol]].dropna(subset=[vcol, mcol])
        tmp = tmp.sort_values(by=[vcol, dcol])
        last = tmp.groupby(vcol, dropna=False).tail(1).rename(columns={mcol: "ultimo_motorista"})
        last = last[[vcol, "ultimo_motorista"]]
        agg = agg.merge(last, on=vcol, how="left")
    else:
        agg["ultimo_motorista"] = None

    total_veiculos = len(agg)
    media_motoristas = agg["qtd_motoristas"].mean() if total_veiculos else 0
    apenas_um = int((agg["qtd_motoristas"] == 1).sum())
    dois_ou_mais = int((agg["qtd_motoristas"] >= 2).sum())
    if not agg.empty:
        top_row = agg.sort_values("qtd_motoristas", ascending=False).iloc[0]
        veic_top = str(top_row[vcol])
        top_qtd = int(top_row["qtd_motoristas"])
    else:
        veic_top, top_qtd = "-", 0

    st.markdown("## 🔄 Rotatividade Motoristas x Veículos")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Veículos com dados", _kpi_value(total_veiculos))
    c2.metric("Média de motoristas/veículo", f"{media_motoristas:.2f}".replace(".", ","))
    c3.metric("Veículos (1 motorista)", _kpi_value(apenas_um))
    c4.metric("Veículos (2+ motoristas)", _kpi_value(dois_ou_mais))
    c5.metric("Maior rotatividade", f"{_kpi_value(top_qtd)} (veíc. {veic_top})")

    st.markdown("—")
    colA, colB, colC = st.columns([1,1,2])
    min_qtd = colA.number_input("Filtrar por quantidade mínima de motoristas", min_value=1, value=1, step=1, key="rot_min_qtd")
    top_n   = colB.number_input("Top N veículos (0 = todos)", min_value=0, value=0, step=5, key="rot_top_n")
    ordenar = colC.selectbox("Ordenar por", ["qtd_motoristas (desc)", "qtd_motoristas (asc)", vcol], key="rot_ordenar")

    data = agg.copy()
    data = data[data["qtd_motoristas"] >= min_qtd]
    if ordenar == "qtd_motoristas (asc)":
        data = data.sort_values("qtd_motoristas", ascending=True)
    elif ordenar == vcol:
        data = data.sort_values(vcol, ascending=True)
    else:
        data = data.sort_values("qtd_motoristas", ascending=False)

    if top_n and top_n > 0:
        data_plot = data.head(int(top_n))
    else:
        data_plot = data

    if not data_plot.empty:
        # Garantir eixo Veículo como descrição
        data_plot[vcol] = data_plot[vcol].astype(str)

        fig = px.bar(
            data_plot,
            x=vcol,
            y="qtd_motoristas",
            title="Quantidade de motoristas únicos por veículo (período selecionado)",
            text="qtd_motoristas",
        )
        fig.update_xaxes(type='category')
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Veículo", yaxis_title="Qtde de motoristas", bargap=0.2, height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum veículo atende ao filtro atual.")

    st.markdown("#### Detalhamento")
    table = data.copy().rename(columns={
        vcol: "Veículo",
        "qtd_motoristas": "Qtde de motoristas",
        "trocas_est": "Trocas (estimadas)",
        "ultimo_motorista": "Último motorista"
    })
    table["Motoristas (lista)"] = table["lista_motoristas"].apply(lambda xs: ", ".join(map(str, xs))[:2000])
    table = table.drop(columns=["lista_motoristas"], errors="ignore")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv = table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV (rotatividade por veículo)", data=csv, file_name="rotatividade_motoristas_por_veiculo.csv", mime="text/csv")
# === Fim painel rotatividade ===




# === Chamada do painel de Rotatividade (injetado) ===
try:
    _df_candidates = [
        'df_scope','df_filtrado','df_filtered','df_periodo','df_period','df_view','df_final','df_result','df_base_filtrado'
    ]
    df_rot = None
    for _name in _df_candidates:
        if _name in globals():
            _obj = globals()[_name]
            try:
                import pandas as _pd
                if isinstance(_obj, _pd.DataFrame) and not _obj.empty:
                    df_rot = _obj
                    break
            except Exception:
                pass
    if df_rot is None and 'df' in globals():
        df_rot = df

    if df_rot is not None:
        show_rotatividade_motoristas_por_veiculo(df_rot)
    else:
        st.info("Não foi possível encontrar o DataFrame filtrado. Ajuste o nome da variável ao chamar o painel.")
except Exception as e:
    st.warning(f"Falha ao renderizar painel de rotatividade: {e}")
# === Fim chamada painel rotatividade ===


# === Painel: Linha do tempo de alocação (1 dia) — COM INDICADORES ===
def show_linha_do_tempo_alocacao_1dia(df, titulo="📆 Linha do tempo de alocação (1 dia)"):
    import pandas as pd
    import plotly.express as px
    from datetime import date as _date

    vcol = "Numero Veiculo"
    lcol = "Nome Linha"
    scol = "Data Hora Inicio Operacao"
    ecol = "Data Hora Final Operacao"

    missing = [c for c in [vcol, lcol, scol, ecol] if c not in df.columns]
    if missing:
        st.error("Colunas ausentes para o painel de alocação (1 dia): " + ", ".join(missing))
        return

    CAND_PASS_TOTAL = ["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros","Total Passageiros","Passageiros Transportados","Qtd de Passageiros","Quantidade de Passageiros"]
    CAND_PAGANTES   = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte","Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes","Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT       = ["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade","Gratuidades","Gratuidade","Quantidade Gratuidade"]

    def _num_from_row(row, cols):
        total = 0.0
        for c in cols:
            if c in row.index:
                try:
                    v = pd.to_numeric(row[c], errors="coerce")
                    if pd.notna(v):
                        total += float(v)
                except Exception:
                    pass
        return float(total)

    def _passageiros_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        pag = _num_from_row(row, CAND_PAGANTES)
        grat = _num_from_row(row, CAND_GRAT)
        return float(pag + grat)

    st.markdown("## " + titulo)

    df = df.copy()
    df[scol] = pd.to_datetime(df[scol], errors="coerce")
    df[ecol] = pd.to_datetime(df[ecol], errors="coerce")

    sdates = df[scol].dropna().dt.date
    edates = df[ecol].dropna().dt.date
    day_default = sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())

    dia = st.date_input("Dia para análise (apenas 1 dia)", value=day_default, format="DD/MM/YYYY", key="aloc_dia")

    day_start = pd.Timestamp(dia).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + pd.Timedelta(days=1)

    pass_cols = [c for c in (CAND_PASS_TOTAL + CAND_PAGANTES + CAND_GRAT) if c in df.columns]
    tmp = df[[vcol, lcol, scol, ecol] + pass_cols].dropna(subset=[scol, ecol, vcol]).copy()

    segs = []
    for _, r in tmp.iterrows():
        s = r[scol]; e = r[ecol]
        if pd.isna(s) or pd.isna(e) or e <= day_start or s >= day_end:
            continue
        s_clip = max(s, day_start)
        e_clip = min(e, day_end)
        if s_clip >= e_clip:
            continue
        pax = _passageiros_row(r)
        segs.append({"Veículo": str(r[vcol]), "Linha": str(r[lcol]), "Início": s_clip, "Fim": e_clip, "Passageiros": pax})

    seg = pd.DataFrame(segs)
    if seg.empty:
        st.info("Sem segmentos para o dia selecionado.")
        return

    seg = seg.sort_values(["Veículo", "Início", "Fim"])
    ociosos = []
    for veic, g in seg.groupby("Veículo", sort=False):
        cur = day_start
        for _, rr in g.iterrows():
            if rr["Início"] > cur:
                ociosos.append({"Veículo": veic, "Linha": "Ocioso", "Início": cur, "Fim": rr["Início"], "Passageiros": 0.0})
            cur = max(cur, rr["Fim"])
        if cur < day_end:
            ociosos.append({"Veículo": veic, "Linha": "Ocioso", "Início": cur, "Fim": day_end, "Passageiros": 0.0})
    if ociosos:
        seg = pd.concat([seg, pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Veículo","Início"])

    seg["Duração (min)"] = (seg["Fim"] - seg["Início"]).dt.total_seconds()/60.0

    with st.expander("Filtros de exibição"):
        seg["Veículo"] = seg["Veículo"].astype(str)
        veics = sorted(seg["Veículo"].unique().tolist())
        linhas = sorted(seg["Linha"].astype(str).unique().tolist())
        pick_veics = st.multiselect("Filtrar Veículos", veics, default=veics, key="aloc_filt_veic")
        pick_linhas = st.multiselect("Filtrar Linhas (inclui 'Ocioso')", linhas, default=linhas, key="aloc_filt_lin")
        segf = seg[(seg["Veículo"].isin(pick_veics)) & (seg["Linha"].astype(str).isin(pick_linhas))]
    if segf.empty:
        st.info("Os filtros atuais não retornaram segmentos.")
        return
    # === Indicadores (Veículo × Linha) ===
    segf = segf.copy()
    segf["_dur_min"] = (segf["Fim"] - segf["Início"]).dt.total_seconds()/60.0
    work_mask = segf["Linha"].astype(str) != "Ocioso"
    t_work = float(segf.loc[work_mask, "_dur_min"].sum())
    t_idle = float(segf.loc[~work_mask, "_dur_min"].sum())
    t_tot = t_work + t_idle
    pax_tot = float(pd.to_numeric(segf.loc[work_mask, "Passageiros"], errors="coerce").fillna(0).sum())
    pph = (pax_tot / (t_work/60.0)) if t_work > 0 else 0.0

    def _fmt_hhmm(m):
        try:
            m = int(round(float(m)))
        except Exception:
            m = 0
        return f"{m//60:02d}:{m%60:02d}"

    def _fmt_pct(x):
        try:
            return f"{(float(x)*100):.1f}%"
        except Exception:
            return "0.0%"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Veículos (selecionados)", f"{segf['Veículo'].nunique()}")
    c2.metric("Tempo Operacional", _fmt_hhmm(t_work))
    c3.metric("Tempo Ocioso", _fmt_hhmm(t_idle))
    c4.metric("% Ocioso", _fmt_pct(t_idle / t_tot) if t_tot>0 else "0.0%")
    c5.metric("Passageiros / Hora", f"{pph:.1f}")
    st.divider()


    # --- Ordenação cronológica por início de viagem (eixo Y: Veículo) ---
    try:
        _order_veiculo = (segf.groupby('Veículo')['Início'].min().sort_values().index.tolist())
        _order_veiculo_plot = list(reversed(_order_veiculo))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_veiculo_plot = list(_pd.unique(segf['Veículo']))
        except Exception:
            _order_veiculo_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---
    fig = px.timeline(
        segf,
        x_start="Início",
        x_end="Fim",
        y="Veículo", category_orders={'Veículo': _order_veiculo_plot}, 
        color="Linha",
        pattern_shape="ZeroPass" if "ZeroPass" in segf.columns else None,
        pattern_shape_map={True: "x", False: ""} if "ZeroPass" in segf.columns else None,
        hover_data={"Duração (min)":":.1f","Passageiros":True,"Veículo":True,"Linha":True,"Início":True,"Fim":True},
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Veículo")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segmentos gerados")
    st.dataframe(segf, use_container_width=True, hide_index=True)
    csv = segf.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV da linha do tempo (1 dia)", data=csv, file_name="alocacao_veiculos_1dia.csv", mime="text/csv")
# === Fim Painel: Linha do tempo de alocação (1 dia) — COM INDICADORES ===



# === Painel: Linha do tempo — Motoristas × Linhas (1 dia) — COM INDICADORES ===
def show_linha_do_tempo_motoristas_linhas_1dia(df, titulo="📆 Linha do tempo: Motoristas × Linhas (1 dia)"):
    import pandas as pd
    import plotly.express as px
    from datetime import date as _date

    vcol = "Numero Veiculo"
    lcol = "Nome Linha"
    scol = "Data Hora Inicio Operacao"
    ecol = "Data Hora Final Operacao"
    M_CANDS = ["Motorista","Operador","Cobrador/Operador","MOTORISTA","Matricula","Matrícula","CPF Motorista","ID Motorista","Nome Motorista","Nome do Motorista"]
    mcol = next((c for c in M_CANDS if c in df.columns), None)

    missing = [c for c in [mcol, lcol, scol, ecol] if c is None or c not in df.columns]
    if missing:
        st.error("Colunas ausentes para o painel Motoristas × Linhas: " + ", ".join(map(str, missing)))
        return

    CAND_PASS_TOTAL = ["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros","Total Passageiros","Passageiros Transportados","Qtd de Passageiros","Quantidade de Passageiros"]
    CAND_PAGANTES   = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte","Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes","Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT       = ["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade","Gratuidades","Gratuidade","Quantidade Gratuidade"]

    def _num_from_row(row, cols):
        total = 0.0
        for c in cols:
            if c in row.index:
                try:
                    v = pd.to_numeric(row[c], errors="coerce")
                    if pd.notna(v):
                        total += float(v)
                except Exception:
                    pass
        return float(total)

    def _passageiros_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        pag = _num_from_row(row, CAND_PAGANTES)
        grat = _num_from_row(row, CAND_GRAT)
        return float(pag + grat)

    st.markdown("## " + titulo)

    df = df.copy()
    df[scol] = pd.to_datetime(df[scol], errors="coerce")
    df[ecol] = pd.to_datetime(df[ecol], errors="coerce")

    sdates = df[scol].dropna().dt.date
    edates = df[ecol].dropna().dt.date
    day_default = sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())

    dia = st.date_input("Dia (1 dia) — Motoristas × Linhas", value=day_default, format="DD/MM/YYYY", key="mot_lin_dia")

    day_start = pd.Timestamp(dia).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + pd.Timedelta(days=1)

    pass_cols = [c for c in (CAND_PASS_TOTAL + CAND_PAGANTES + CAND_GRAT) if c in df.columns]
    tmp = df[[mcol, lcol, scol, ecol] + pass_cols].dropna(subset=[mcol, scol, ecol]).copy()

    segs = []
    for _, r in tmp.iterrows():
        s = r[scol]; e = r[ecol]
        if pd.isna(s) or pd.isna(e) or e <= day_start or s >= day_end:
            continue
        s_clip = max(s, day_start)
        e_clip = min(e, day_end)
        if s_clip >= e_clip:
            continue
        pax = _passageiros_row(r)
        segs.append({"Motorista": str(r[mcol]), "Linha": str(r[lcol]), "Início": s_clip, "Fim": e_clip, "Passageiros": pax})

    seg = pd.DataFrame(segs)
    if seg.empty:
        st.info("Sem segmentos para o dia selecionado.")
        return

    seg = seg.sort_values(["Motorista", "Início", "Fim"])
    ociosos = []
    for mot, g in seg.groupby("Motorista", sort=False):
        cur = day_start
        for _, rr in g.iterrows():
            if rr["Início"] > cur:
                ociosos.append({"Motorista": mot, "Linha": "Ocioso", "Início": cur, "Fim": rr["Início"], "Passageiros": 0.0})
            cur = max(cur, rr["Fim"])
        if cur < day_end:
            ociosos.append({"Motorista": mot, "Linha": "Ocioso", "Início": cur, "Fim": day_end, "Passageiros": 0.0})
    if ociosos:
        seg = pd.concat([seg, pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Motorista","Início"])

    seg["Duração (min)"] = (seg["Fim"] - seg["Início"]).dt.total_seconds()/60.0

    
    # === Totais por motorista e rótulos com HE ===
    def _fmt_hhmm(total_min):
        try:
            total_min = int(round(float(total_min)))
        except Exception:
            total_min = 0
        h = total_min // 60
        m = total_min % 60
        return f"{h:02d}:{m:02d}"

    _mask_work = seg["Linha"].astype(str) != "Ocioso"
    # usa a duração já calculada, zerando quando ocioso
    _work_min = seg["Duração (min)"].where(_mask_work, 0.0)
    _totais = seg.assign(__work_min=_work_min).groupby("Motorista", observed=False)["__work_min"].sum(min_count=1).fillna(0.0)

    _limite_min = 7*60 + 20  # 07:20
    mot_label_map = {}
    
    for _mot, _mins in _totais.items():
        _extra = max(0, _mins - _limite_min)
        if _extra > 0:
            mot_label_map[_mot] = f"⚡ <b>{_mot} — {_fmt_hhmm(_mins)} (HE {_fmt_hhmm(_extra)})</b>"
        else:
            mot_label_map[_mot] = f"{_mot} — {_fmt_hhmm(_mins)}"
    with st.expander("Filtros — Motoristas × Linhas"):
        mot_list = sorted(seg["Motorista"].astype(str).unique().tolist())
        linhas = sorted(seg["Linha"].astype(str).unique().tolist())
        pick_mot = st.multiselect("Filtrar Motoristas", mot_list, default=mot_list, key="ml_filt_mot")
        pick_lin = st.multiselect("Filtrar Linhas (inclui 'Ocioso')", linhas, default=linhas, key="ml_filt_lin")
        segf = seg[(seg["Motorista"].isin(pick_mot)) & (seg["Linha"].astype(str).isin(pick_lin))]
    if segf.empty:
        st.info("Os filtros atuais não retornaram segmentos.")
        return

    # Flag ZeroPass (somente não-ocioso)
    import pandas as _pd
    segf["ZeroPass"] = (segf["Linha"].astype(str) != "Ocioso") & (_pd.to_numeric(segf["Passageiros"], errors="coerce").fillna(-1) == 0)
    # === Indicadores (Motoristas × Linhas) ===
    segf = segf.copy()
    segf["_dur_min"] = (segf["Fim"] - segf["Início"]).dt.total_seconds()/60.0
    work_mask = segf["Linha"].astype(str) != "Ocioso"
    t_work = float(segf.loc[work_mask, "_dur_min"].sum())
    pax_tot = float(pd.to_numeric(segf.loc[work_mask, "Passageiros"], errors="coerce").fillna(0).sum())
    mot_count = int(segf["Motorista"].nunique())
    work_per_mot = segf.loc[work_mask].groupby("Motorista", observed=False)["_dur_min"].sum() if mot_count>0 else None
    he_total = float((work_per_mot - (7*60 + 20)).clip(lower=0).sum()) if work_per_mot is not None else 0.0
    avg_work_min = (t_work / mot_count) if mot_count>0 else 0.0
    pph = (pax_tot / (t_work/60.0)) if t_work > 0 else 0.0

    def _fmt_hhmm(m):
        try:
            m = int(round(float(m)))
        except Exception:
            m = 0
        return f"{m//60:02d}:{m%60:02d}"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Motoristas (selecionados)", f"{mot_count}")
    c2.metric("Horas trabalhadas (total)", _fmt_hhmm(t_work))
    c3.metric("Hora extra (total)", _fmt_hhmm(he_total))
    c4.metric("Média h/ motorista", _fmt_hhmm(avg_work_min))
    c5.metric("Passageiros / Hora", f"{pph:.1f}")
    st.divider()

    # Label opcional já existente em versões anteriores permanece válido
    # Rótulo do eixo Y com horas e HE
    segf['Motorista_Label'] = segf['Motorista'].map(mot_label_map).fillna(segf['Motorista'].astype(str))



    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---
    fig = px.timeline(
        segf,
        x_start="Início",
        x_end="Fim",
        y='Motorista_Label', category_orders={'Motorista_Label': _order_motorista_plot}, 
        color="Linha",
        pattern_shape="ZeroPass" if "ZeroPass" in segf.columns else None,
        pattern_shape_map={True: "x", False: ""} if "ZeroPass" in segf.columns else None,
        hover_data={"Duração (min)":":.1f","Passageiros":True,"ZeroPass":True,"Motorista":True,"Linha":True,"Início":True,"Fim":True},
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Motorista")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segmentos — Motoristas × Linhas")
    st.dataframe(segf, use_container_width=True, hide_index=True)
    csv = segf.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV (motoristas × linhas - 1 dia)", data=csv, file_name="motoristas_x_linhas_1dia.csv", mime="text/csv")
# === Fim Painel: Motoristas × Linhas (1 dia) — COM INDICADORES ===



# === Painel: Linha do tempo — Motoristas × Veículos (1 dia) — COM INDICADORES ===
def show_linha_do_tempo_motoristas_veiculos_1dia(df, titulo="📆 Linha do tempo: Motoristas × Veículos (1 dia)"):
    import pandas as pd
    import plotly.express as px
    from datetime import date as _date

    vcol = "Numero Veiculo"
    scol = "Data Hora Inicio Operacao"
    ecol = "Data Hora Final Operacao"
    M_CANDS = ["Motorista","Operador","Cobrador/Operador","MOTORISTA","Matricula","Matrícula","CPF Motorista","ID Motorista","Nome Motorista","Nome do Motorista"]
    mcol = next((c for c in M_CANDS if c in df.columns), None)

    missing = [c for c in [mcol, vcol, scol, ecol] if c is None or c not in df.columns]
    if missing:
        st.error("Colunas ausentes para o painel Motoristas × Veículos: " + ", ".join(map(str, missing)))
        return

    CAND_PASS_TOTAL = ["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros","Total Passageiros","Passageiros Transportados","Qtd de Passageiros","Quantidade de Passageiros"]
    CAND_PAGANTES   = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte","Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes","Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT       = ["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade","Gratuidades","Gratuidade","Quantidade Gratuidade"]

    def _num_from_row(row, cols):
        total = 0.0
        for c in cols:
            if c in row.index:
                try:
                    v = pd.to_numeric(row[c], errors="coerce")
                    if pd.notna(v):
                        total += float(v)
                except Exception:
                    pass
        return float(total)

    def _passageiros_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        pag = _num_from_row(row, CAND_PAGANTES)
        grat = _num_from_row(row, CAND_GRAT)
        return float(pag + grat)

    st.markdown("## " + titulo)

    df = df.copy()
    df[scol] = pd.to_datetime(df[scol], errors="coerce")
    df[ecol] = pd.to_datetime(df[ecol], errors="coerce")

    sdates = df[scol].dropna().dt.date
    edates = df[ecol].dropna().dt.date
    day_default = sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())

    dia = st.date_input("Dia (1 dia) — Motoristas × Veículos", value=day_default, format="DD/MM/YYYY", key="mot_vei_dia")

    day_start = pd.Timestamp(dia).replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + pd.Timedelta(days=1)

    pass_cols = [c for c in (CAND_PASS_TOTAL + CAND_PAGANTES + CAND_GRAT) if c in df.columns]
    tmp = df[[mcol, vcol, scol, ecol] + pass_cols].dropna(subset=[mcol, vcol, scol, ecol]).copy()

    segs = []
    for _, r in tmp.iterrows():
        s = r[scol]; e = r[ecol]
        if pd.isna(s) or pd.isna(e) or e <= day_start or s >= day_end:
            continue
        s_clip = max(s, day_start)
        e_clip = min(e, day_end)
        if s_clip >= e_clip:
            continue
        pax = _passageiros_row(r)
        segs.append({"Motorista": str(r[mcol]), "Veículo": str(r[vcol]), "Início": s_clip, "Fim": e_clip, "Passageiros": pax})

    seg = pd.DataFrame(segs)
    if seg.empty:
        st.info("Sem segmentos para o dia selecionado.")
        return

    seg = seg.sort_values(["Motorista", "Início", "Fim"])
    ociosos = []
    for mot, g in seg.groupby("Motorista", sort=False):
        cur = day_start
        for _, rr in g.iterrows():
            if rr["Início"] > cur:
                ociosos.append({"Motorista": mot, "Veículo": "Ocioso", "Início": cur, "Fim": rr["Início"], "Passageiros": 0.0})
            cur = max(cur, rr["Fim"])
        if cur < day_end:
            ociosos.append({"Motorista": mot, "Veículo": "Ocioso", "Início": cur, "Fim": day_end, "Passageiros": 0.0})
    if ociosos:
        seg = pd.concat([seg, pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Motorista","Início"])

    seg["Duração (min)"] = (seg["Fim"] - seg["Início"]).dt.total_seconds()/60.0

    
    # === Totais por motorista e rótulos com HE ===
    def _fmt_hhmm(total_min):
        try:
            total_min = int(round(float(total_min)))
        except Exception:
            total_min = 0
        h = total_min // 60
        m = total_min % 60
        return f"{h:02d}:{m:02d}"

    _mask_work = seg["Veículo"].astype(str) != "Ocioso"
    # usa a duração já calculada, zerando quando ocioso
    _work_min = seg["Duração (min)"].where(_mask_work, 0.0)
    _totais = seg.assign(__work_min=_work_min).groupby("Motorista", observed=False)["__work_min"].sum(min_count=1).fillna(0.0)

    _limite_min = 7*60 + 20  # 07:20
    mot_label_map = {}
    
    for _mot, _mins in _totais.items():
        _extra = max(0, _mins - _limite_min)
        if _extra > 0:
            mot_label_map[_mot] = f"⚡ <b>{_mot} — {_fmt_hhmm(_mins)} (HE {_fmt_hhmm(_extra)})</b>"
        else:
            mot_label_map[_mot] = f"{_mot} — {_fmt_hhmm(_mins)}"
    with st.expander("Filtros — Motoristas × Veículos"):
        mot_list = sorted(seg["Motorista"].astype(str).unique().tolist())
        veics = sorted(seg["Veículo"].astype(str).unique().tolist())
        pick_mot = st.multiselect("Filtrar Motoristas", mot_list, default=mot_list, key="mv_filt_mot")
        pick_vei = st.multiselect("Filtrar Veículos (inclui 'Ocioso')", veics, default=veics, key="mv_filt_vei")
        segf = seg[(seg["Motorista"].isin(pick_mot)) & (seg["Veículo"].astype(str).isin(pick_vei))]
    if segf.empty:
        st.info("Os filtros atuais não retornaram segmentos.")
        return

    # Flag ZeroPass (somente não-ocioso)
    import pandas as _pd
    segf["ZeroPass"] = (segf["Veículo"].astype(str) != "Ocioso") & (_pd.to_numeric(segf["Passageiros"], errors="coerce").fillna(-1) == 0)
    # === Indicadores (Motoristas × Veículos) ===
    segf = segf.copy()
    segf["_dur_min"] = (segf["Fim"] - segf["Início"]).dt.total_seconds()/60.0
    work_mask = segf["Veículo"].astype(str) != "Ocioso"
    t_work = float(segf.loc[work_mask, "_dur_min"].sum())
    pax_tot = float(pd.to_numeric(segf.loc[work_mask, "Passageiros"], errors="coerce").fillna(0).sum())
    mot_count = int(segf["Motorista"].nunique())
    work_per_mot = segf.loc[work_mask].groupby("Motorista", observed=False)["_dur_min"].sum() if mot_count>0 else None
    he_total = float((work_per_mot - (7*60 + 20)).clip(lower=0).sum()) if work_per_mot is not None else 0.0
    avg_work_min = (t_work / mot_count) if mot_count>0 else 0.0
    pph = (pax_tot / (t_work/60.0)) if t_work > 0 else 0.0

    def _fmt_hhmm(m):
        try:
            m = int(round(float(m)))
        except Exception:
            m = 0
        return f"{m//60:02d}:{m%60:02d}"

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Motoristas (selecionados)", f"{mot_count}")
    c2.metric("Horas trabalhadas (total)", _fmt_hhmm(t_work))
    c3.metric("Hora extra (total)", _fmt_hhmm(he_total))
    c4.metric("Média h/ motorista", _fmt_hhmm(avg_work_min))
    c5.metric("Passageiros / Hora", f"{pph:.1f}")
    st.divider()
    # Rótulo do eixo Y com horas e HE
    segf['Motorista_Label'] = segf['Motorista'].map(mot_label_map).fillna(segf['Motorista'].astype(str))



    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---

    # --- Ordenação cronológica por início de viagem (eixo Y: Motorista_Label) ---
    try:
        _order_motorista = (segf.groupby('Motorista_Label')['Início'].min().sort_values().index.tolist())
        _order_motorista_plot = list(reversed(_order_motorista))  # y-axis será invertido abaixo
    except Exception:
        try:
            import pandas as _pd
            _order_motorista_plot = list(_pd.unique(segf['Motorista_Label']))
        except Exception:
            _order_motorista_plot = []
    # Garantir ordem cronológica dos segmentos no eixo X
    try:
        segf = segf.sort_values(by=['Início','Fim'], ascending=[True, True], kind='mergesort')
    except Exception:
        pass
    # --- fim ordenação ---
    fig = px.timeline(
        segf,
        x_start="Início",
        x_end="Fim",
        y='Motorista_Label', category_orders={'Motorista_Label': _order_motorista_plot}, 
        color="Veículo",
        pattern_shape="ZeroPass" if "ZeroPass" in segf.columns else None,
        pattern_shape_map={True: "x", False: ""} if "ZeroPass" in segf.columns else None,
        hover_data={"Duração (min)":":.1f","Passageiros":True,"ZeroPass":True,"Motorista":True,"Veículo":True,"Início":True,"Fim":True},
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Motorista")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segmentos — Motoristas × Veículos")
    st.dataframe(segf, use_container_width=True, hide_index=True)
    csv = segf.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV (motoristas × veículos - 1 dia)", data=csv, file_name="motoristas_x_veiculos_1dia.csv", mime="text/csv")
# === Fim Painel: Motoristas × Veículos (1 dia) — COM INDICADORES ===



# === Chamada: Linha do tempo de alocação (1 dia) ===
try:
    if 'show_linha_do_tempo_alocacao_1dia' in globals():
        _df_candidates = [
            'df_scope','df_filtrado','df_filtered','df_periodo','df_period','df_view','df_final','df_result','df_base_filtrado','df'
        ]
        _required = ['Numero Veiculo','Nome Linha','Data Hora Inicio Operacao','Data Hora Final Operacao']
        df_candidate = None
        for _name in _df_candidates:
            if _name in globals():
                _obj = globals()[_name]
                try:
                    import pandas as _pd
                    if isinstance(_obj, _pd.DataFrame) and not _obj.empty:
                        if set(_required).issubset(set(_obj.columns)):
                            df_candidate = _obj
                            break
                except Exception:
                    pass
        if df_candidate is not None:
            show_linha_do_tempo_alocacao_1dia(df_candidate)
except Exception as e:
    st.warning(f"Falha ao renderizar painel de alocação (1 dia): {e}")
# === Fim chamada: Linha do tempo de alocação (1 dia) ===

# === Chamada: Timeline Motoristas × Linhas (1 dia) ===
try:
    if 'show_linha_do_tempo_motoristas_linhas_1dia' in globals():
        _df_candidates = [
            'df_scope','df_filtrado','df_filtered','df_periodo','df_period','df_view','df_final','df_result','df_base_filtrado','df'
        ]
        _required = ['Nome Linha','Data Hora Inicio Operacao','Data Hora Final Operacao']
        df_candidate = None
        for _name in _df_candidates:
            if _name in globals():
                _obj = globals()[_name]
                try:
                    import pandas as _pd
                    if isinstance(_obj, _pd.DataFrame) and not _obj.empty:
                        if set(_required).issubset(set(_obj.columns)):
                            df_candidate = _obj
                            break
                except Exception:
                    pass
        if df_candidate is not None:
            show_linha_do_tempo_motoristas_linhas_1dia(df_candidate)
except Exception as e:
    st.warning(f"Falha ao renderizar painel Motoristas × Linhas: {e}")
# === Fim chamada: Timeline Motoristas × Linhas (1 dia) ===

# === Chamada: Timeline Motoristas × Veículos (1 dia) ===
try:
    if 'show_linha_do_tempo_motoristas_veiculos_1dia' in globals():
        _df_candidates = [
            'df_scope','df_filtrado','df_filtered','df_periodo','df_period','df_view','df_final','df_result','df_base_filtrado','df'
        ]
        _required = ['Numero Veiculo','Data Hora Inicio Operacao','Data Hora Final Operacao']
        df_candidate = None
        for _name in _df_candidates:
            if _name in globals():
                _obj = globals()[_name]
                try:
                    import pandas as _pd
                    if isinstance(_obj, _pd.DataFrame) and not _obj.empty:
                        if set(_required).issubset(set(_obj.columns)):
                            df_candidate = _obj
                            break
                except Exception:
                    pass
        if df_candidate is not None:
            show_linha_do_tempo_motoristas_veiculos_1dia(df_candidate)
except Exception as e:
    st.warning(f"Falha ao renderizar painel Motoristas × Veículos: {e}")
# === Fim chamada: Timeline Motoristas × Veículos (1 dia) ===
