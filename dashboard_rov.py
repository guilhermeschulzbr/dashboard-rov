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
from pandas.api.types import is_datetime64_any_dtype as is_dt64
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

def load_json_config(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
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
def load_data(csv_path: str) -> pd.DataFrame:
    """Carrega o CSV (sep=';'), normaliza tipos e deriva colunas úteis."""
    encodings = ["utf-8", "latin-1"]
    last_err = None
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, sep=";", encoding=enc)
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
BASE_DATE_FOR_FALLBACK = pd.Timestamp("2025-08-01")

def load_km_store() -> dict:
    return load_json_config(CONFIG_PATH_KM)

def save_km_store(store: dict) -> bool:
    return save_json_config(CONFIG_PATH_KM, store)

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
    """Retorna km vigente na data 'when'; se não houver, tenta a vigência válida em 01/08/2025."""
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
    return load_json_config(CONFIG_PATH_VEIC)

def save_veic_store(store: dict) -> bool:
    return save_json_config(CONFIG_PATH_VEIC, store)

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
DEFAULT_PATH = os.path.join(os.getcwd(), "dados_ROV.csv")
st.sidebar.title("⚙️ Configurações")
csv_path = st.sidebar.text_input("Arquivo de dados (CSV ';')", DEFAULT_PATH)
if not os.path.exists(csv_path):
    st.error(f"Arquivo não encontrado: {csv_path}")
    st.stop()

with st.spinner("Carregando dados..."):
    df = load_data(csv_path)

st.title("📊 Dashboard Operacional ROV")
st.caption("*Baseado exclusivamente nas colunas existentes do arquivo `dados_ROV.csv`*")

# ------------------------------
# Classificação de Linhas (Urbana/Distrital) com persistência
# ------------------------------
if "Nome Linha" in df.columns:
    st.sidebar.header("Classificação de Linhas")
    cfg_categ = load_json_config(CONFIG_PATH_CATEG)
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
                if save_json_config(CONFIG_PATH_CATEG, novo_cfg):
                    st.success("Classificação salva.")
                    cfg_categ = novo_cfg
        with col_s2:
            if st.button("↩️ Reset (limpar)", use_container_width=True):
                if save_json_config(CONFIG_PATH_CATEG, {}):
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

# Renderiza o novo topo de KPIs (2x3)
try:
    _render_top_kpis(df_filtered if 'df_filtered' in globals() else df)
except Exception as _e:
    try:
        st.warning(f"Falha ao renderizar KPIs do topo: {_e}")
    except Exception:
        pass


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
    cat_opt = st.sidebar.selectbox("Categoria", options=["Todas", "Urbanas", "Distritais"], index=0)
    if cat_opt != "Todas":
        df_filtered = df_filtered[df_filtered["Categoria Linha"] == ("Urbana" if cat_opt == "Urbanas" else "Distrital")]

# Veículo
if "Numero Veiculo" in df_filtered.columns:
    veics = sorted([str(x) for x in df_filtered["Numero Veiculo"].dropna().astype(str).unique().tolist()])
    sel_veics = st.sidebar.multiselect("Veículos", veics)
    if sel_veics:
        df_filtered = df_filtered[df_filtered["Numero Veiculo"].astype(str).isin(sel_veics)]

# Terminal
# Expurgo / Visualização
st.sidebar.header("Expurgo de viagens")
expurgar_zero = st.sidebar.checkbox("Expurgar viagens com 0 passageiros", value=False)
expurgar_trein = st.sidebar.checkbox("Expurgar motoristas em treinamento", value=False)
modo_visu = st.sidebar.selectbox("Modo de visualização", options=["Normal", "Apenas expurgados"], index=0)

# Como identificar "treinamento"
possiveis_col_status = [c for c in ["Descricao Tipo Evento","Tipo Viagem","Observacao","Grupo Veiculo","Categoria Linha","Cobrador/Operador","Matricula"] if c in df_filtered.columns]
col_status = st.sidebar.selectbox("Coluna para identificar treinamento", options=["(automático)"] + possiveis_col_status, index=0)
palavras_trein = st.sidebar.text_input("Palavras-chave (separadas por vírgula)", value="trein, treinamento")

def make_training_mask(df_in):
    if col_status != "(automático)":
        serie = df_in[col_status].astype(str).str.lower()
        keys = [p.strip().lower() for p in palavras_trein.split(",") if p.strip()]
        if not keys:
            return pd.Series(False, index=df_in.index)
        pat = "|".join([re.escape(k) for k in keys])
        return serie.str.contains(pat, na=False)
    else:
        # tenta em colunas possíveis
        keys = [p.strip().lower() for p in palavras_trein.split(",") if p.strip()]
        if not keys:
            return pd.Series(False, index=df_in.index)
        pat = "|".join([re.escape(k) for k in keys])
        for cc in possiveis_col_status:
            if df_in[cc].astype(str).str.contains(pat, case=False, na=False).any():
                return df_in[cc].astype(str).str.contains(pat, case=False, na=False)
        return pd.Series(False, index=df_in.index)

mask_zero = pd.Series(False, index=df_filtered.index)
if "Passageiros" in df_filtered.columns:
    mask_zero = df_filtered["Passageiros"].fillna(0) <= 0

mask_trein = make_training_mask(df_filtered) if expurgar_trein or modo_visu=="Apenas expurgados" else pd.Series(False, index=df_filtered.index)

if modo_visu == "Apenas expurgados":
    # Se nenhum critério marcado, não filtra
    criterio = pd.Series(False, index=df_filtered.index)
    if expurgar_zero:
        criterio = criterio | mask_zero
    if expurgar_trein:
        criterio = criterio | mask_trein
    if criterio.any():
        df_filtered = df_filtered[criterio]
    else:
        st.sidebar.info("Ative pelos menos um critério para visualizar apenas expurgados.")
else:
    # Modo normal: removemos os expurgados marcados
    if expurgar_zero:
        df_filtered = df_filtered[~mask_zero]
    if expurgar_trein:
        df_filtered = df_filtered[~mask_trein]

if "Descricao Terminal" in df_filtered.columns:
    terms = sorted([x for x in df_filtered["Descricao Terminal"].dropna().unique().tolist()])
    sel_terms = st.sidebar.multiselect("Terminais", terms)
    if sel_terms:
        df_filtered = df_filtered[df_filtered["Descricao Terminal"].isin(sel_terms)]

# Parâmetros de alerta
st.sidebar.header("Alertas")
thr_dist_alta = st.sidebar.number_input("Distância alta (km) ≥", min_value=0.0, value=20.0, step=1.0, format="%.0f")
thr_pax_baixa = st.sidebar.number_input("Passageiros baixos ≤", min_value=0, value=5, step=1, format="%d")

# Parâmetros financeiros
st.sidebar.header("Financeiro")
tarifa_usuario = st.sidebar.number_input("Tarifa ao usuário (R$)", min_value=0.0, value=2.00, step=0.10, format="%.2f")
subsidio_pagante = st.sidebar.number_input("Subsídio por pagante (R$)", min_value=0.0, value=4.20, step=0.10, format="%.2f")

# ------------------------------
# KPIs
# ------------------------------
kpi_cols = st.columns(6)

# Passageiros total
total_pax = df_filtered["Passageiros"].sum() if "Passageiros" in df_filtered.columns else 0
kpi_cols[0].metric("👥 Passageiros", fmt_int(total_pax))

# Viagens registradas
viagens = len(df_filtered)
kpi_cols[1].metric("🧭 Viagens registradas", fmt_int(viagens))

# Distância total (usa distância configurada quando existir)
if "Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any():
    dist_total = df_filtered["Distancia_cfg_km"].sum(min_count=1)
else:
    dist_total = df_filtered["Distancia"].sum() if "Distancia" in df_filtered.columns else 0.0
kpi_cols[2].metric("🛣️ Distância total (km)", fmt_float(dist_total, 1))

# Média pax/viagem
media_pax = (total_pax / viagens) if viagens > 0 else 0.0
kpi_cols[3].metric("📈 Média pax/viagem", fmt_float(media_pax, 2))

# Veículos (IDs distintos na base filtrada)
veics_ids = df_filtered["Numero Veiculo"].nunique() if "Numero Veiculo" in df_filtered.columns else 0
kpi_cols[4].metric("🚌 Veículos (IDs distintos)", fmt_int(veics_ids))

# Linhas ativas
linhas_ativas = df_filtered["Nome Linha"].nunique() if "Nome Linha" in df_filtered.columns else 0
kpi_cols[5].metric("🧵 Linhas ativas", fmt_int(linhas_ativas))

# --- Financeiro (com base nas colunas existentes) ---
paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
integration_cols_all = ["Quant Passagem Integracao","Quant Passe Integracao","Quant Vale Transporte Integracao"]
present_paying = [c for c in paying_cols_all if c in df_filtered.columns]
present_integration = [c for c in integration_cols_all if c in df_filtered.columns]

total_pagantes = float(df_filtered[present_paying].sum().sum()) if present_paying else 0.0
total_integracoes = float(df_filtered[present_integration].sum().sum()) if present_integration else 0.0
total_gratuidade = float(df_filtered["Quant Gratuidade"].sum()) if "Quant Gratuidade" in df_filtered.columns else 0.0

receita_tarifaria = total_pagantes * float(tarifa_usuario)
subsidio_total = total_pagantes * float(subsidio_pagante)
receita_total = receita_tarifaria + subsidio_total

pax_total_calc = float(total_pax) if pd.notna(total_pax) else 0.0
custo_publico_por_pax_total = (subsidio_total / pax_total_calc) if pax_total_calc > 0 else 0.0

st.subheader("💰 Indicadores financeiros (parâmetros na barra lateral)")
fin_cols = st.columns(7)
fin_cols[0].metric("Pagantes", fmt_int(total_pagantes))
fin_cols[1].metric("Integrações (sem custo)", fmt_int(total_integracoes))
fin_cols[2].metric("Gratuidades", fmt_int(total_gratuidade))
fin_cols[3].metric("Receita tarifária", fmt_currency(receita_tarifaria, 2))
fin_cols[4].metric("Subsídio total", fmt_currency(subsidio_total, 2))
fin_cols[5].metric("Custo público/pax", fmt_currency(custo_publico_por_pax_total, 2))
fin_cols[6].metric("Receita total", fmt_currency(receita_total, 2))

# ---------- NOVOS INDICADORES ----------
# IPK (passageiros por km)
ipk_total    = (total_pax / dist_total)     if dist_total and dist_total > 0 else 0.0
ipk_pagantes = (total_pagantes / dist_total) if dist_total and dist_total > 0 else 0.0

# Receita por km rodado (usando receita total)
receita_por_km = (receita_total / dist_total) if dist_total and dist_total > 0 else 0.0

# Veículos configurados médios no período: média por linha e soma
if "Veiculos_cfg" in df_filtered.columns and df_filtered["Veiculos_cfg"].notna().any():
    veic_cfg_por_linha = df_filtered.groupby("Nome Linha", observed=False)["Veiculos_cfg"].mean(numeric_only=True)
    veic_cfg_total_medio = veic_cfg_por_linha.sum()
else:
    veic_cfg_total_medio = 0.0

# Receita por veículo configurado
receita_por_veic_cfg = (receita_total / veic_cfg_total_medio) if veic_cfg_total_medio and veic_cfg_total_medio > 0 else 0.0

# Médias por veículo configurado
viagens_por_veic = (viagens / veic_cfg_total_medio) if veic_cfg_total_medio and veic_cfg_total_medio > 0 else 0.0
km_por_veic      = (dist_total / veic_cfg_total_medio) if veic_cfg_total_medio and veic_cfg_total_medio > 0 else 0.0
pax_por_veic     = (total_pax / veic_cfg_total_medio)  if veic_cfg_total_medio and veic_cfg_total_medio > 0 else 0.0

# Bloco de KPIs adicionais
st.subheader("🚀 Indicadores avançados")
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("IPK total (pax/km)", fmt_float(ipk_total, 3))
colB.metric("IPK pagantes (pax/km)", fmt_float(ipk_pagantes, 3))
colC.metric("Receita por km", fmt_currency(receita_por_km, 2))
colD.metric("Veículos configurados (média)", fmt_float(veic_cfg_total_medio, 2))
colE.metric("Receita por veículo", fmt_currency(receita_por_veic_cfg, 2))

# KPIs médios por veículo
colF, colG, colH = st.columns(3)
colF.metric("Viagens por veículo", fmt_float(viagens_por_veic, 2))
colG.metric("KM por veículo", fmt_float(km_por_veic, 2))
colH.metric("Passageiros por veículo", fmt_float(pax_por_veic, 2))

# ------------------------------
# Gráficos
# ------------------------------


# ------------------------------
# Indicadores por motorista
# ------------------------------
st.subheader("🧑‍✈️ Indicadores por motorista")

motorista_col = None
for cc in ["Cobrador/Operador", "Matricula"]:
    if cc in df_filtered.columns:
        motorista_col = cc
        break

if motorista_col is None:
    st.info("Não encontrei colunas de motorista (ex.: 'Cobrador/Operador' ou 'Matricula').")
else:
    # Agregações por motorista
    dist_col_drv = "Distancia_cfg_km" if ("Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in df_filtered.columns else None)
    grp_m = df_filtered.groupby(motorista_col, observed=False)

    viagens_m = grp_m.size().rename("Viagens")
    pax_m = grp_m["Passageiros"].sum(numeric_only=True) if "Passageiros" in df_filtered.columns else pd.Series(0, index=viagens_m.index)
    km_m = grp_m[dist_col_drv].sum(numeric_only=True) if dist_col_drv else pd.Series(0.0, index=viagens_m.index)

    paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    present_paying_m = [c for c in paying_cols_all if c in df_filtered.columns]
    if present_paying_m:
        pag_df = grp_m[present_paying_m].sum(numeric_only=True)
        pag_m = pag_df.sum(axis=1)
    else:
        pag_m = pd.Series(0.0, index=viagens_m.index)

    receita_m = pag_m * (float(tarifa_usuario) + float(subsidio_pagante))

    # ---------- NOVO: Aproveitamento (horas trabalhadas vs carga diária 7:20) ----------
    REF_HOURS = 7 + 20/60  # 7h20 = 7.333...
    start_col = "Data Hora Inicio Operacao" if "Data Hora Inicio Operacao" in df_filtered.columns else None
    end_col   = "Data Hora Final Operacao" if "Data Hora Final Operacao" in df_filtered.columns else None

    util_df = pd.DataFrame()
    if start_col and end_col:
        tmp = df_filtered[[motorista_col, start_col, end_col]].copy()
        tmp["dur_h"] = (pd.to_datetime(tmp[end_col], errors="coerce") - pd.to_datetime(tmp[start_col], errors="coerce")).dt.total_seconds() / 3600.0
        # remove negativos/zeros
        tmp.loc[(tmp["dur_h"] <= 0) | ~np.isfinite(tmp["dur_h"]), "dur_h"] = np.nan
        # dia de referência (do início); fallback para Data Coleta
        if "Data Coleta" in df_filtered.columns:
            tmp["dia_ref"] = pd.to_datetime(df_filtered["Data Coleta"], errors="coerce").dt.date
        else:
            tmp["dia_ref"] = pd.to_datetime(df_filtered[start_col], errors="coerce").dt.date
        util_grp = tmp.dropna(subset=["dur_h"]).groupby([motorista_col, "dia_ref"], observed=False)["dur_h"].sum().reset_index()
        horas_por_motorista = util_grp.groupby(motorista_col, observed=False)["dur_h"].sum()
        dias_por_motorista = util_grp.groupby(motorista_col, observed=False)["dia_ref"].nunique()
        aproveitamento_pct = horas_por_motorista / (dias_por_motorista * REF_HOURS)
        aproveitamento_pct = aproveitamento_pct.replace([np.inf, -np.inf], np.nan)
    else:
        horas_por_motorista = pd.Series(dtype=float)
        dias_por_motorista = pd.Series(dtype=float)
        aproveitamento_pct = pd.Series(dtype=float)

    # KPIs médios por motorista (sobre o conjunto filtrado)
    n_motoristas = len(viagens_m.index)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Motoristas distintos", fmt_int(n_motoristas))
    k2.metric("Média de viagens/motorista", fmt_float((viagens / n_motoristas) if n_motoristas else 0, 2))
    k3.metric("Média de pax/motorista", fmt_float((total_pax / n_motoristas) if n_motoristas else 0, 2))
    k4.metric("Média de km/motorista", fmt_float((dist_total / n_motoristas) if n_motoristas else 0, 2))
    k5.metric("Média de receita/motorista", fmt_currency((receita_total / n_motoristas) if n_motoristas else 0, 2))

    # Bloco de aproveitamento agregado
    st.markdown("**Aproveitamento (horas trabalhadas ÷ 7:20 por dia)**")
    if not aproveitamento_pct.empty:
        media_aprov = float(np.nanmean(aproveitamento_pct.values)) if len(aproveitamento_pct) else np.nan
        pct_full = float(np.mean(aproveitamento_pct >= 1.0)) if len(aproveitamento_pct) else 0.0
        pct_baixo = float(np.mean(aproveitamento_pct <= 0.8)) if len(aproveitamento_pct) else 0.0
        a1, a2, a3 = st.columns(3)
        a1.metric("Média de aproveitamento", fmt_pct(media_aprov, 1))
        a2.metric("% motoristas ≥ 100%", fmt_pct(pct_full, 1))
        a3.metric("% motoristas ≤ 80%", fmt_pct(pct_baixo, 1))
    else:
        st.info("Para calcular o aproveitamento, são necessários 'Data Hora Inicio Operacao' e 'Data Hora Final Operacao'.")

    # Rankings
    st.markdown("**Rankings por motorista (Top 20)**")
    colM1, colM2, colM3, colM4, colM5 = st.columns(5)

    with colM1:
        top_pax = pd.DataFrame({"Motorista": pax_m.index, "Passageiros": pax_m.values}).sort_values("Passageiros", ascending=False).head(20)
        top_pax["Passageiros"] = top_pax["Passageiros"].apply(fmt_int)
        st.caption("Mais passageiros")
        st.dataframe(top_pax, use_container_width=True)

    with colM2:
        top_viag = pd.DataFrame({"Motorista": viagens_m.index, "Viagens": viagens_m.values}).sort_values("Viagens", ascending=False).head(20)
        top_viag["Viagens"] = top_viag["Viagens"].apply(fmt_int)
        st.caption("Mais viagens")
        st.dataframe(top_viag, use_container_width=True)

    with colM3:
        top_km = pd.DataFrame({"Motorista": km_m.index, "KM": km_m.values}).sort_values("KM", ascending=False).head(20)
        top_km["KM"] = top_km["KM"].apply(lambda v: fmt_float(v, 1))
        st.caption("Mais KM")
        st.dataframe(top_km, use_container_width=True)

    with colM4:
        top_rec = pd.DataFrame({"Motorista": receita_m.index, "Receita": receita_m.values}).sort_values("Receita", ascending=False).head(20)
        top_rec["Receita"] = top_rec["Receita"].apply(lambda v: fmt_currency(v, 2))
        st.caption("Mais receita")
        st.dataframe(top_rec, use_container_width=True)

    with colM5:
        if not aproveitamento_pct.empty:
            top_aprov = aproveitamento_pct.sort_values(ascending=False).head(20).reset_index()
            top_aprov.columns = ["Motorista", "Aproveitamento"]
            top_aprov["Aproveitamento"] = top_aprov["Aproveitamento"].apply(lambda v: fmt_pct(v, 1))
            st.caption("Maior aproveitamento")
            st.dataframe(top_aprov, use_container_width=True)
        else:
            st.caption("Maior aproveitamento")
            st.info("Sem dados suficientes para o ranking de aproveitamento.")


# ------------------------------
# Motoristas com menores valores (Bottom 20)
# ------------------------------
st.markdown("**Motoristas com menores valores (Bottom 20)**")
colB1, colB2, colB3, colB4, colB5 = st.columns(5)

with colB1:
    bot_pax = pd.DataFrame({"Motorista": pax_m.index, "Passageiros": pax_m.fillna(0).values}).sort_values("Passageiros", ascending=True).head(20)
    bot_pax["Passageiros"] = bot_pax["Passageiros"].apply(fmt_int)
    st.caption("Menos passageiros")
    st.dataframe(bot_pax, use_container_width=True)

with colB2:
    bot_viag = pd.DataFrame({"Motorista": viagens_m.index, "Viagens": pd.Series(viagens_m).fillna(0).values}).sort_values("Viagens", ascending=True).head(20)
    bot_viag["Viagens"] = bot_viag["Viagens"].apply(fmt_int)
    st.caption("Menos viagens")
    st.dataframe(bot_viag, use_container_width=True)

with colB3:
    _km_series = pd.Series(km_m).fillna(0)
    bot_km = pd.DataFrame({"Motorista": _km_series.index, "KM": _km_series.values}).sort_values("KM", ascending=True).head(20)
    bot_km["KM"] = bot_km["KM"].apply(lambda v: fmt_float(v, 1))
    st.caption("Menos KM")
    st.dataframe(bot_km, use_container_width=True)

with colB4:
    _rec_series = pd.Series(receita_m).fillna(0)
    bot_rec = pd.DataFrame({"Motorista": _rec_series.index, "Receita": _rec_series.values}).sort_values("Receita", ascending=True).head(20)
    bot_rec["Receita"] = bot_rec["Receita"].apply(lambda v: fmt_currency(v, 2))
    st.caption("Menos receita")
    st.dataframe(bot_rec, use_container_width=True)

with colB5:
    if not aproveitamento_pct.empty:
        _ap_series = pd.Series(aproveitamento_pct).fillna(0)
        bot_aprov = _ap_series.sort_values(ascending=True).head(20).reset_index()
        bot_aprov.columns = ["Motorista", "Aproveitamento"]
        bot_aprov["Aproveitamento"] = bot_aprov["Aproveitamento"].apply(lambda v: fmt_pct(v, 1))
        st.caption("Menor aproveitamento")
        st.dataframe(bot_aprov, use_container_width=True)
    else:
        st.caption("Menor aproveitamento")
        st.info("Sem dados suficientes para o ranking de aproveitamento.")


# ------------------------------
# 🔎 Detalhes do motorista (seleção consolidada)
# ------------------------------
try:
    candidates_set = set()
    for varname in ["top_pax","top_viag","top_km","top_rec","top_aprov","bot_pax","bot_viag","bot_km","bot_rec","bot_aprov"]:
        if varname in locals():
            df_tmp = locals()[varname]
            if isinstance(df_tmp, pd.DataFrame) and "Motorista" in df_tmp.columns:
                candidates_set.update([str(x) for x in df_tmp["Motorista"].dropna().astype(str).tolist()])
    candidates = sorted(list(candidates_set))
    sel_any = select_motorista_widget(candidates, key="sel_any_motorista", label="🔎 Ver detalhes (Top/Bottom)")
    if sel_any and sel_any != "(selecione)":
        show_motorista_details(sel_any, df_filtered)
except Exception as _e:
    pass


# ------------------------------
# IA (Beta)
# ------------------------------
st.sidebar.header("🤖 IA (Beta)")

ai_perf = st.sidebar.checkbox("Score de performance de motoristas (ajustado por contexto)", value=False, help="Compara o que o motorista entregou vs o esperado para o contexto da viagem.")
ai_cluster = st.sidebar.checkbox("Clusterização de linhas (K-Means)", value=False, help="Agrupa linhas por perfil operacional.")
k_clusters = st.sidebar.slider("Clusters (linhas)", min_value=2, max_value=8, value=4, step=1)

ai_anom = st.sidebar.checkbox("Detecção de anomalias (IsolationForest)", value=False, help="Identifica viagens atípicas com base em múltiplos sinais.")
ai_fore = st.sidebar.checkbox("Previsão de demanda por linha (Prophet)", value=False, help="Prevê passageiros por dia/linha para planejamento.")
contam = st.sidebar.slider("Anomalias: fração esperada (%)", min_value=1, max_value=10, value=3, step=1, help="Percentual aproximado de outliers no conjunto.", format="%d%%")
forecast_horizon = st.sidebar.number_input("Previsão: horizonte (dias)", min_value=7, max_value=90, value=30, step=1)
linha_opts = ["(auto)"]
if "Nome Linha" in df_filtered.columns:
    linha_opts += sorted([x for x in df_filtered["Nome Linha"].dropna().astype(str).unique().tolist()])
line_for_forecast = st.sidebar.selectbox("Linha para previsão", options=linha_opts, index=0)

# ---------- Detecção de anomalias ----------
anomias_df = pd.DataFrame()
if ai_anom:
    st.subheader("⚠️ Detecção de anomalias (IsolationForest)")
    if not _HAS_SKLEARN:
        st.error("scikit-learn não encontrado. Instale com: `pip install scikit-learn`")
    else:
        # Montagem de features
        base = df_filtered.copy()
        # Coluna de distância
        dist_col_x = "Distancia_cfg_km" if ("Distancia_cfg_km" in base.columns and base["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in base.columns else None)
        # Duração
        if {"Data Hora Inicio Operacao","Data Hora Final Operacao"}.issubset(base.columns):
            di = pd.to_datetime(base["Data Hora Inicio Operacao"], errors="coerce")
            df_ = pd.to_datetime(base["Data Hora Final Operacao"], errors="coerce")
            dur_min = (df_ - di).dt.total_seconds() / 60.0
        else:
            dur_min = pd.Series(np.nan, index=base.index)
        # Hora/Dia
        hora_b = base["Hora_Base"] if "Hora_Base" in base.columns else pd.Series(np.nan, index=base.index)
        dow_b  = base["DiaSemana_Base"] if "DiaSemana_Base" in base.columns else pd.Series(np.nan, index=base.index)
        # Quantitativos
        pax = base["Passageiros"] if "Passageiros" in base.columns else pd.Series(np.nan, index=base.index)
        grat = base["Quant Gratuidade"] if "Quant Gratuidade" in base.columns else pd.Series(0.0, index=base.index)
        paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
        present_paying_loc = [c for c in paying_cols_all if c in base.columns]
        pag = base[present_paying_loc].sum(axis=1) if present_paying_loc else pd.Series(0.0, index=base.index)
        veic_cfg = base["Veiculos_cfg"] if "Veiculos_cfg" in base.columns else pd.Series(np.nan, index=base.index)
        # Encodes simples para linha e motorista (opcionais)
        lin_code = base["Nome Linha"].cat.codes if "Nome Linha" in base.columns and str(base["Nome Linha"].dtype).startswith("category") else pd.Categorical(base["Nome Linha"]).codes if "Nome Linha" in base.columns else pd.Series(np.nan, index=base.index)
        mot_col = "Cobrador/Operador" if "Cobrador/Operador" in base.columns else ("Matricula" if "Matricula" in base.columns else None)
        mot_code = (pd.Categorical(base[mot_col]).codes if mot_col else pd.Series(np.nan, index=base.index))

        feats = pd.DataFrame({
            "pax": pax,
            "km": base[dist_col_x] if dist_col_x else np.nan,
            "dur_min": dur_min,
            "hora": hora_b,
            "dow": dow_b,
            "grat": grat,
            "pag": pag,
            "veic_cfg": veic_cfg,
            "lin": lin_code,
            "mot": mot_code,
        })
        feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
        if len(feats) < 20:
            st.info("Dados insuficientes para treinar o modelo de anomalias (mín. 20 linhas com features completas).")
        else:
            iso = IsolationForest(
                n_estimators=200,
                contamination=contam/100.0,
                random_state=42,
            )
            iso.fit(feats.values)
            pred = iso.predict(feats.values)        # -1 = anômalo, 1 = normal
            score = -iso.decision_function(feats.values)  # maior = mais anômalo
            anom_mask = (pred == -1)
            idx_rows = feats.index
            base_sub = base.loc[idx_rows].copy()
            base_sub["anomalia"] = anom_mask
            base_sub["score_anomalia"] = pd.Series(score, index=idx_rows)
            anomias_df = base_sub[base_sub["anomalia"]].sort_values("score_anomalia", ascending=False)

            # Gráfico: Passageiros x Distância (ou duração), colorindo anomalias
            xcol = dist_col_x if dist_col_x else "dur_min"
            if xcol in base_sub.columns or xcol == "dur_min":
                x_series = base_sub[xcol] if xcol in base_sub.columns else feats["dur_min"]
                graf = pd.DataFrame({
                    "x": x_series,
                    "Passageiros": base_sub["Passageiros"] if "Passageiros" in base_sub.columns else np.nan,
                    "anom": base_sub["anomalia"],
                })
                fig = px.scatter(graf, x="x", y="Passageiros", color="anom", title="Anomalias: Passageiros vs " + ("Distância (km)" if xcol==dist_col_x else "Duração (min)"))
                fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=380, legend_title_text="Anômalo?")
                st.plotly_chart(fig, use_container_width=True)

            st.caption("Top anomalias (ordenadas por severidade)")
            cols_show = [c for c in ["Data Coleta","Nome Linha","Numero Veiculo","Descricao Terminal","Passageiros"] if c in anomias_df.columns]
            if dist_col_x and dist_col_x in anomias_df.columns:
                cols_show += [dist_col_x]
            if "score_anomalia" in anomias_df.columns:
                cols_show += ["score_anomalia"]
            preview = anomias_df[cols_show].head(100).copy()
            if "Passageiros" in preview.columns:
                preview["Passageiros"] = preview["Passageiros"].apply(fmt_int)
            if dist_col_x and dist_col_x in preview.columns:
                preview[dist_col_x] = preview[dist_col_x].apply(lambda v: fmt_float(v, 1))
            if "score_anomalia" in preview.columns:
                preview["score_anomalia"] = preview["score_anomalia"].apply(lambda v: fmt_float(v, 3))
            st.dataframe(preview, use_container_width=True)

            # Exportação
            @st.cache_data(show_spinner=False)
            def _anom_csv(df_in: pd.DataFrame) -> bytes:
                return df_in.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
            st.download_button("Baixar anomalias (CSV ;)", data=_anom_csv(anomias_df), file_name="anomalias_viagens.csv", mime="text/csv")


# ---------- Score de performance de motoristas (ajustado por contexto) ----------
if ai_perf:
    st.subheader("🏁 Performance de motoristas (ajustada por contexto)")
    if not _HAS_SKLEARN:
        st.error("scikit-learn não encontrado. Instale com: `pip install scikit-learn`")
    else:
        base = df_filtered.copy()
        # Requisitos mínimos
        need_cols = ["Passageiros"]
        if not all(c in base.columns for c in need_cols):
            st.info("É necessário ter a coluna 'Passageiros' para calcular o score.")
        else:
            # Variáveis de contexto
            # Distância
            dist_col = "Distancia_cfg_km" if ("Distancia_cfg_km" in base.columns and base["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in base.columns else None)
            if dist_col is None:
                base["__km__"] = np.nan
                dist_col = "__km__"
            # Duração (min)
            if {"Data Hora Inicio Operacao","Data Hora Final Operacao"}.issubset(base.columns):
                di = pd.to_datetime(base["Data Hora Inicio Operacao"], errors="coerce")
                df_ = pd.to_datetime(base["Data Hora Final Operacao"], errors="coerce")
                base["dur_min"] = (df_ - di).dt.total_seconds() / 60.0
            else:
                base["dur_min"] = np.nan
            # Hora/Dia
            base["hora"] = base["Hora_Base"] if "Hora_Base" in base.columns else np.nan
            base["dow"]  = base["DiaSemana_Base"] if "DiaSemana_Base" in base.columns else np.nan
            # Veículos configurados
            base["veic_cfg_x"] = base["Veiculos_cfg"] if "Veiculos_cfg" in base.columns else np.nan
            # Categoria (urb/distr)
            base["cat_lin"] = base["Categoria Linha"].astype(str) if "Categoria Linha" in base.columns else "NA"
            # Linha
            base["lin"] = base["Nome Linha"].astype(str) if "Nome Linha" in base.columns else "NA"
            # Motorista
            mot_col = "Cobrador/Operador" if "Cobrador/Operador" in base.columns else ("Matricula" if "Matricula" in base.columns else None)
            base["motorista"] = base[mot_col].astype(str) if mot_col else "NA"

            # Monta df de treino com dummies simples
            feat_cols = [dist_col, "dur_min", "hora", "dow", "veic_cfg_x"]
            cat_cols = []
            if "cat_lin" in base.columns: cat_cols.append("cat_lin")
            if "lin" in base.columns: cat_cols.append("lin")

            df_model = base[["Passageiros"] + feat_cols + cat_cols + ["motorista"]].copy()
            df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna(subset=["Passageiros"] + feat_cols)

            # Limita nº de dummies para linhas (evita explosão)
            if "lin" in cat_cols:
                # mantém top 30 linhas e agrupa resto em 'OUTRAS'
                top_linhas = base["lin"].value_counts().head(30).index
                df_model["lin_grp"] = np.where(df_model["lin"].isin(top_linhas), df_model["lin"], "OUTRAS")
                cat_cols_use = [c for c in cat_cols if c != "lin"] + ["lin_grp"]
            else:
                cat_cols_use = cat_cols

            X = pd.get_dummies(df_model[feat_cols + cat_cols_use], drop_first=True)
            y = df_model["Passageiros"].astype(float)

            # Remove colunas vazias/constantes
            X = X.loc[:, X.std(numeric_only=True) > 0]

            if len(X) < 50 or X.isna().any().any():
                st.info("Dados insuficientes/limpos para treinar o modelo de baseline.")
            else:
                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    resid = y - y_pred  # positivo = acima do esperado

                    df_model = df_model.assign(y_pred=y_pred, resid=resid)
                    # Agrega por motorista
                    perf = df_model.groupby("motorista", observed=False).agg(
                        viagens=("motorista","size"),
                        pax_real=("Passageiros","sum"),
                        pax_prev=("y_pred","sum"),
                        resid_med=("resid","mean"),
                        resid_tot=("resid","sum")
                    ).reset_index().sort_values("resid_tot", ascending=False)

                    # Mostra rankings
                    c1, c2 = st.columns(2)
                    top_perf = perf.head(20).copy()
                    top_perf["resid_tot"] = top_perf["resid_tot"].apply(fmt_int)
                    top_perf["resid_med"] = top_perf["resid_med"].apply(lambda v: fmt_float(v,2))
                    top_perf["pax_real"] = top_perf["pax_real"].apply(fmt_int)
                    top_perf["pax_prev"] = top_perf["pax_prev"].apply(fmt_int)
                    c1.caption("Maior valor agregado (acima do esperado)")
                    c1.dataframe(top_perf, use_container_width=True)

                    bot_perf = perf.tail(20).copy().sort_values("resid_tot", ascending=True)
                    bot_perf["resid_tot"] = bot_perf["resid_tot"].apply(fmt_int)
                    bot_perf["resid_med"] = bot_perf["resid_med"].apply(lambda v: fmt_float(v,2))
                    bot_perf["pax_real"] = bot_perf["pax_real"].apply(fmt_int)
                    bot_perf["pax_prev"] = bot_perf["pax_prev"].apply(fmt_int)
                    c2.caption("Menor valor agregado (abaixo do esperado)")
                    c2.dataframe(bot_perf, use_container_width=True)

                    # Selecionar motorista para detalhes de performance
                    cand = perf["motorista"].astype(str).tolist()
                    selp = st.selectbox("🔎 Ver detalhes de performance (ajustada)", options=["(selecione)"] + cand, index=0, key="sel_perf_motorista")
                    if selp and selp != "(selecione)":
                        dfm = df_model[df_model["motorista"].astype(str) == str(selp)]
                        if not dfm.empty:
                            st.markdown(f"**Detalhe do motorista {selp}**")
                            # Série: real vs previsto por data (se houver)
                            if "Data" in base.columns and base["Data"].notna().any():
                                base_join = base[["motorista","Data","Passageiros"]].copy()
                                base_join = base_join[base_join["motorista"].astype(str)==str(selp)]
                                # agrega por dia
                                ser_real = base_join.groupby("Data", as_index=False, observed=False)["Passageiros"].sum()
                                # previsto: usar média do resid por viagem daquele dia (aproximação simples)
                                st.dataframe(ser_real.head(50))
                            # Distribuição dos resíduos
                            import numpy as _np
                            import plotly.express as _px
                            try:
                                figh = _px.histogram(dfm, x="resid", nbins=30, title="Distribuição dos resíduos (real - previsto)")
                                st.plotly_chart(figh, use_container_width=True)
                            except Exception:
                                pass
                except Exception as e:
                    st.error(f"Falha no cálculo de performance ajustada: {e}")

# ---------- Previsão de demanda por linha ----------
if ai_fore:
    st.subheader("📈 Previsão de passageiros por linha (Prophet)")
    if not _HAS_PROPHET:
        st.error("Prophet não encontrado. Instale com: `pip install prophet`")
    else:
        # Seleção de linhas a prever
        dff = df_filtered.copy()
        if "Data" not in dff.columns or "Passageiros" not in dff.columns or not dff["Data"].notna().any():
            st.info("São necessários 'Data' e 'Passageiros' agregáveis para a previsão.")
        else:
            # Linhas candidatas
            linhas_disp = sorted([x for x in dff["Nome Linha"].dropna().astype(str).unique().tolist()]) if "Nome Linha" in dff.columns else []
            if line_for_forecast != "(auto)" and line_for_forecast in linhas_disp:
                linhas_alvo = [line_for_forecast]
            else:
                # Escolhe até 6 linhas com maior volume recente
                top = (dff.groupby("Nome Linha", observed=False)["Passageiros"].sum().sort_values(ascending=False).head(6).index.tolist()) if "Nome Linha" in dff.columns else []
                linhas_alvo = top

            if not linhas_alvo:
                st.info("Nenhuma linha encontrada para previsão.")
            else:
                for ln in linhas_alvo:
                    st.markdown(f"**Linha: {ln}**")
                    serie = (dff[dff["Nome Linha"] == ln]
                             .groupby("Data", as_index=False, observed=False)["Passageiros"].sum()
                             .sort_values("Data"))
                    serie = serie.dropna(subset=["Data","Passageiros"])
                    if len(serie) < 14:
                        st.info("Histórico insuficiente para previsão desta linha (mín. 14 dias).")
                        continue
                    ds = pd.to_datetime(serie["Data"])
                    y = serie["Passageiros"].astype(float)
                    model_df = pd.DataFrame({"ds": ds, "y": y})
                    try:
                        m = Prophet(seasonality_mode="multiplicative", weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
                        m.fit(model_df)
                        fut = m.make_future_dataframe(periods=int(forecast_horizon), freq="D", include_history=True)
                        fc = m.predict(fut)
                        # Junta real x previsto
                        fc_vis = fc[["ds","yhat","yhat_lower","yhat_upper"]].merge(model_df, on="ds", how="left")
                        fc_vis.sort_values("ds", inplace=True)
                        # Métrica de insight: média últimos 7 dias reais vs próximos 7 previstos
                        ult7 = model_df.set_index("ds")["y"].last("7D").mean() if not model_df.empty else np.nan
                        prox7 = fc.set_index("ds")["yhat"].last("7D").mean() if not fc.empty else np.nan
                        delta = (prox7 - ult7) / ult7 if (pd.notna(ult7) and ult7 not in (0, np.nan)) else np.nan

                        # Gráfico
                        figf = px.line(fc_vis, x="ds", y=["y","yhat"], labels={"value":"Passageiros","ds":"Data","variable":"Série"}, title=f"Previsão (horizonte {int(forecast_horizon)} dias)")
                        figf.add_scatter(x=fc_vis["ds"], y=fc_vis["yhat_upper"], mode="lines", name="limite superior", line=dict(dash="dot"))
                        figf.add_scatter(x=fc_vis["ds"], y=fc_vis["yhat_lower"], mode="lines", name="limite inferior", line=dict(dash="dot"))
                        figf.update_xaxes(tickformat="%d/%m/%Y")
                        figf.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=380)
                        st.plotly_chart(figf, use_container_width=True)

                        # KPIs de insight
                        cA, cB = st.columns(2)
                        cA.metric("Média últimos 7 dias (real)", fmt_int(ult7 if pd.notna(ult7) else 0))
                        cB.metric("Variação esperada (próx. 7 dias)", fmt_pct(delta if pd.notna(delta) else 0, 1))
                    except Exception as e:
                        st.error(f"Falha ao ajustar Prophet para a linha {ln}: {e}")

st.subheader("📅 Evolução temporal de passageiros")
if {"Data", "Passageiros"}.issubset(df_filtered.columns) and df_filtered["Data"].notna().any():
    serie = df_filtered.groupby("Data", as_index=False, observed=False)["Passageiros"].sum().sort_values("Data")
    fig = px.line(serie, x="Data", y="Passageiros", markers=True, title="Passageiros por dia")
    fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=350)
    fig.update_xaxes(tickformat="%d/%m/%Y")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados de 'Data Coleta' e 'Passageiros' suficientes para a série temporal.")

left, right = st.columns(2)

with left:
    st.subheader("🏅 Ranking de linhas por demanda")
    if {"Nome Linha", "Passageiros"}.issubset(df_filtered.columns):
        rank = (df_filtered.groupby("Nome Linha", as_index=False, observed=False)["Passageiros"].sum()
                .sort_values("Passageiros", ascending=False).head(15))
        # Formatação PT-BR com separador de milhares
        rank_fmt = rank.copy()
        rank_fmt["Passageiros"] = rank_fmt["Passageiros"].apply(fmt_int)
        fig = px.bar(rank, x="Nome Linha", y="Passageiros", title="Top 15 linhas por passageiros")
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10,r=10,t=35,b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Tabela (formatada)")
        st.dataframe(rank_fmt.head(50))
    else:
        st.info("Colunas necessárias ausentes: 'Nome Linha' e/ou 'Passageiros'.")

with right:
    st.subheader("🧾 Distribuição por tipo de tarifa")
    tarifa_cols = [c for c in df_filtered.columns if c.startswith("Quant ") or c in ["Quant Inteiras"]]
    tarifa_cols = [c for c in tarifa_cols if df_filtered[c].notna().any()]
    if tarifa_cols:
        totais = df_filtered[tarifa_cols].sum().reset_index()
        totais.columns = ["Tipo", "Quantidade"]
        totais = totais[totais["Quantidade"] > 0]
        if not totais.empty:
            fig = px.pie(totais, names="Tipo", values="Quantidade", hole=0.45, title="Participação por tipo")
            fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("As colunas de tarifa existem, mas não há valores positivos no filtro atual.")
    else:
        st.info("Não foram encontradas colunas de tarifação (Quant * / Quant Inteiras) com dados.")

st.subheader("📍 Relação distância x passageiros")
x_col = "Distancia_cfg_km" if ("Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in df_filtered.columns else None)
if x_col and {x_col, "Passageiros"}.issubset(df_filtered.columns) and df_filtered[[x_col,"Passageiros"]].notna().any().any():
    trendline_kw = {}
    trendline_note = ""
    try:
        import statsmodels.api as sm  # noqa: F401
        trendline_kw = {"trendline": "ols"}
    except Exception:
        trendline_note = " (sem linha de tendência — instale 'statsmodels' para habilitar)"
    fig = px.scatter(
        df_filtered,
        x=x_col,
        y="Passageiros",
        hover_data=[c for c in ["Nome Linha","Numero Veiculo","Descricao Terminal","Data Coleta"] if c in df_filtered.columns],
        title=f"Dispersão: {'Distância configurada' if x_col=='Distancia_cfg_km' else 'Distância de viagem'} (km) x Passageiros" + trendline_note,
        **trendline_kw
    )
    fig.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados suficientes para plotar distância x passageiros.")

st.subheader("⏰ Picos por hora e dia da semana")
if {"Passageiros"}.issubset(df_filtered.columns):
    if ("Hora_Base" not in df_filtered.columns) or (not df_filtered["Hora_Base"].notna().any()):
        for c in ["Data Hora Saida Terminal", "Data Hora Inicio Operacao", "Data Coleta"]:
            if c in df_filtered.columns:
                temp_dt = pd.to_datetime(df_filtered[c], errors="coerce")
                if temp_dt.notna().any():
                    df_filtered["Hora_Base"] = temp_dt.dt.hour
                    df_filtered["DiaSemana_Base"] = temp_dt.dt.dayofweek
                    break
if {"Hora_Base","DiaSemana_Base","Passageiros"}.issubset(df_filtered.columns) and df_filtered["Hora_Base"].notna().any():
    heat = (df_filtered.dropna(subset=["Hora_Base","DiaSemana_Base"])
            .groupby(["DiaSemana_Base","Hora_Base"], as_index=False, observed=False)["Passageiros"].sum())
    if not heat.empty:
        heat["DiaSemana_Label"] = heat["DiaSemana_Base"].map({0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"})
        fig = px.density_heatmap(
            heat,
            x="Hora_Base", y="DiaSemana_Label", z="Passageiros",
            histfunc="avg", nbinsx=24, title="Heatmap de demanda (hora x dia)"
        )
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Não há dados suficientes para o heatmap no filtro atual.")
else:
    st.info("Sem colunas de horário adequadas para gerar o heatmap.")

# ------------------------------
# Alertas
# ------------------------------
st.subheader("🚨 Alertas operacionais (baseado apenas nas colunas existentes)")
alertas = []

# Sem passageiros
if "Passageiros" in df_filtered.columns:
    z = df_filtered[df_filtered["Passageiros"].fillna(0) <= 0]
    if not z.empty:
        alertas.append(("Viagens sem passageiros", z))

# Distância zero (considera distância configurada, senão a original)
base_x = "Distancia_cfg_km" if ("Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in df_filtered.columns else None)
if base_x:
    dz = df_filtered[df_filtered[base_x].fillna(0) <= 0]
    if not dz.empty:
        alertas.append(("Viagens com distância zero", dz))

# Distância alta & baixa demanda
if base_x and {base_x, "Passageiros"}.issubset(df_filtered.columns):
    cond = (df_filtered[base_x] >= thr_dist_alta) & (df_filtered["Passageiros"] <= thr_pax_baixa)
    db = df_filtered[cond]
    if not db.empty:
        alertas.append((f"Distância ≥ {thr_dist_alta} km e Passageiros ≤ {thr_pax_baixa}", db))

if alertas:
    for titulo, df_a in alertas:
        st.markdown(f"**• {titulo}: {len(df_a)} registros**")
        with st.expander("Ver amostra"):
            st.dataframe(df_a.head(100))
else:
    st.success("Nenhum alerta gerado com os filtros atuais.")

# ------------------------------
# Resumos rápidos (com formatação PT-BR)
# ------------------------------
st.subheader("📑 Resumos rápidos")
col_a, col_b = st.columns(2)

with col_a:
    if {"Nome Linha","Passageiros"}.issubset(df_filtered.columns):
        g = (df_filtered.groupby("Nome Linha", as_index=False, observed=False)["Passageiros"].sum()
             .sort_values("Passageiros", ascending=False))
        st.caption("Passageiros por linha")
        st.dataframe(df_fmt_milhar(g, ["Passageiros"]).head(50))

        # Receita por linha (pagantes)
        if present_paying:
            by_line = df_filtered.groupby("Nome Linha", as_index=False, observed=False)[present_paying].sum()
            by_line["Pagantes"] = by_line[present_paying].sum(axis=1)
            by_line["Receita_tarifaria_R$"] = by_line["Pagantes"] * float(tarifa_usuario)
            by_line["Subsídio_R$"] = by_line["Pagantes"] * float(subsidio_pagante)
            by_line["Receita_total_R$"] = by_line["Receita_tarifaria_R$"] + by_line["Subsídio_R$"]
            by_line = by_line[["Nome Linha","Pagantes","Receita_tarifaria_R$","Subsídio_R$","Receita_total_R$"]]
            by_line = by_line.sort_values("Receita_total_R$", ascending=False)
            st.caption("Receita por linha (pagantes)")
            st.dataframe(
                df_fmt_currency(
                    df_fmt_milhar(by_line, ["Pagantes"]),
                    ["Receita_tarifaria_R$","Subsídio_R$","Receita_total_R$"]
                ).head(50)
            )

with col_b:
    # Resumo por veículo (IDs distintos reais) e por veículos configurados
    if {"Numero Veiculo"}.issubset(df_filtered.columns):
        dist_col = "Distancia_cfg_km" if ("Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in df_filtered.columns else None)
        agg_dict = {"Passageiros": ("Passageiros","sum")}
        if dist_col:
            agg_dict["Dist_km"] = (dist_col,"sum")
        g2 = (df_filtered.groupby("Numero Veiculo", as_index=False, observed=False)
              .agg(**agg_dict)
              .sort_values("Passageiros", ascending=False if "Passageiros" in df_filtered.columns else True))
        st.caption("Resumo por veículo (IDs reais na base)")
        g2_fmt = g2.copy()
        if "Dist_km" in g2_fmt.columns:
            g2_fmt["Dist_km"] = g2_fmt["Dist_km"].apply(lambda v: fmt_float(v, 1))
        if "Passageiros" in g2_fmt.columns:
            g2_fmt["Passageiros"] = g2_fmt["Passageiros"].apply(fmt_int)
        st.dataframe(g2_fmt.head(50))

# ------------------------------
# Tabela consolidada por linha (métricas completas + cores + totalização)
# ------------------------------
st.subheader("📘 Tabela consolidada por linha")

if "Nome Linha" in df_filtered.columns:
    dist_col_tbl = "Distancia_cfg_km" if ("Distancia_cfg_km" in df_filtered.columns and df_filtered["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in df_filtered.columns else None)
    base_tbl = df_filtered.copy()
    if dist_col_tbl is None:
        base_tbl["__dist__"] = 0.0
        dist_col_tbl = "__dist__"
    grp = base_tbl.groupby("Nome Linha", observed=False)

    veic_cfg_med_tbl  = grp["Veiculos_cfg"].mean(numeric_only=True) if "Veiculos_cfg" in base_tbl.columns else pd.Series(0.0, index=grp.size().index)
    veic_ids_uni_tbl  = grp["Numero Veiculo"].nunique() if "Numero Veiculo" in base_tbl.columns else pd.Series(0, index=grp.size().index)
    km_rodado_tbl     = grp[dist_col_tbl].sum(numeric_only=True)
    pax_total_tbl     = grp["Passageiros"].sum(numeric_only=True) if "Passageiros" in base_tbl.columns else pd.Series(0, index=grp.size().index)
    viagens_total_tbl = grp.size()
    grat_tbl          = grp["Quant Gratuidade"].sum(numeric_only=True) if "Quant Gratuidade" in base_tbl.columns else pd.Series(0.0, index=grp.size().index)

    paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    present_paying_l = [c for c in paying_cols_all if c in base_tbl.columns]
    if present_paying_l:
        pag_by_cols_tbl = grp[present_paying_l].sum(numeric_only=True)
        pagantes_tbl = pag_by_cols_tbl.sum(axis=1)
    else:
        pagantes_tbl = pd.Series(0.0, index=grp.size().index)

    receita_tar_l_tbl = pagantes_tbl * float(tarifa_usuario)
    subsidio_l_tbl    = pagantes_tbl * float(subsidio_pagante)
    receita_tot_l_tbl = receita_tar_l_tbl + subsidio_l_tbl

    pct_grat_s_pag_tbl = grat_tbl / pagantes_tbl.replace(0, np.nan)
    ipk_total_l_tbl    = pax_total_tbl / km_rodado_tbl.replace(0, np.nan)
    ipk_pag_l_tbl      = pagantes_tbl / km_rodado_tbl.replace(0, np.nan)
    rec_por_veic_tbl   = receita_tot_l_tbl / veic_cfg_med_tbl.replace(0, np.nan)
    rec_por_pax_tbl    = receita_tot_l_tbl / pax_total_tbl.replace(0, np.nan)
    rec_por_km_tbl     = receita_tot_l_tbl / km_rodado_tbl.replace(0, np.nan)
    pax_tot_viag_tbl   = pax_total_tbl / viagens_total_tbl.replace(0, np.nan)
    pag_viag_tbl       = pagantes_tbl / viagens_total_tbl.replace(0, np.nan)
    giro_veic_tbl      = veic_ids_uni_tbl / veic_cfg_med_tbl.replace(0, np.nan)  # % de giro

    tabela = pd.DataFrame({
        "Veículos Conf.": veic_cfg_med_tbl,
        "Veículos (IDs Distintos)": veic_ids_uni_tbl,
        "% Giro de Veículos": giro_veic_tbl,
        "Km Rodada": km_rodado_tbl,
        "Pass. Transp.": pax_total_tbl,
        "Pass. Grat.": grat_tbl,
        "Pass. Pag.": pagantes_tbl,
        "% Grat. s/ Pag.": pct_grat_s_pag_tbl,
        "IPK Total": ipk_total_l_tbl,
        "IPK Pag.": ipk_pag_l_tbl,
        "R$ Rec. p/ Veic. Conf.": rec_por_veic_tbl,
        "R$ Rec. p/ Pass Tot.": rec_por_pax_tbl,
        "R$ Rec. p/ Km rodado": rec_por_km_tbl,
        "Pass. Tot. p/ Viagem": pax_tot_viag_tbl,
        "Pass. Pag. p/ Viagem": pag_viag_tbl,
    }).reset_index().rename(columns={"Nome Linha":"Nome Linha"})

    # ===== Totalização =====
    total_veic_cfg_sum   = veic_cfg_med_tbl.sum(skipna=True)
    total_veic_ids_sum   = veic_ids_uni_tbl.sum(skipna=True)
    total_km_sum         = km_rodado_tbl.sum(skipna=True)
    total_pax_sum        = pax_total_tbl.sum(skipna=True)
    total_grat_sum       = grat_tbl.sum(skipna=True)
    total_pag_sum        = pagantes_tbl.sum(skipna=True)
    total_viagens_sum    = viagens_total_tbl.sum(skipna=True)
    total_receita_sum    = receita_tot_l_tbl.sum(skipna=True)

    total_pct_grat_pag   = (total_grat_sum / total_pag_sum) if total_pag_sum else np.nan
    total_ipk_total      = (total_pax_sum / total_km_sum) if total_km_sum else np.nan
    total_ipk_pag        = (total_pag_sum / total_km_sum) if total_km_sum else np.nan
    total_rec_por_veic   = (total_receita_sum / total_veic_cfg_sum) if total_veic_cfg_sum else np.nan
    total_rec_por_pax    = (total_receita_sum / total_pax_sum) if total_pax_sum else np.nan
    total_rec_por_km     = (total_receita_sum / total_km_sum) if total_km_sum else np.nan
    total_pax_tot_viag   = (total_pax_sum / total_viagens_sum) if total_viagens_sum else np.nan
    total_pag_viag       = (total_pag_sum / total_viagens_sum) if total_viagens_sum else np.nan
    total_giro_veic      = (total_veic_ids_sum / total_veic_cfg_sum) if total_veic_cfg_sum else np.nan

    total_row = pd.DataFrame([{
        "Nome Linha": "TOTAL",
        "Veículos Conf.": total_veic_cfg_sum,             # soma das médias por linha
        "Veículos (IDs Distintos)": total_veic_ids_sum,   # soma por linha
        "% Giro de Veículos": total_giro_veic,
        "Km Rodada": total_km_sum,
        "Pass. Transp.": total_pax_sum,
        "Pass. Grat.": total_grat_sum,
        "Pass. Pag.": total_pag_sum,
        "% Grat. s/ Pag.": total_pct_grat_pag,
        "IPK Total": total_ipk_total,
        "IPK Pag.": total_ipk_pag,
        "R$ Rec. p/ Veic. Conf.": total_rec_por_veic,
        "R$ Rec. p/ Pass Tot.": total_rec_por_pax,
        "R$ Rec. p/ Km rodado": total_rec_por_km,
        "Pass. Tot. p/ Viagem": total_pax_tot_viag,
        "Pass. Pag. p/ Viagem": total_pag_viag,
    }])

    tabela = tabela.sort_values(by="R$ Rec. p/ Km rodado", ascending=False, na_position="last")
    tabela = pd.concat([tabela, total_row], ignore_index=True)

    # ===== Cores =====
    # Maior = pior (vermelho): solicitado
    worse_when_high = [
        "Veículos Conf.", "Veículos (IDs Distintos)",
        "% Giro de Veículos", "Km Rodada", "Pass. Grat."
    ]
    # Menor = melhor (já solicitado para % Grat. s/ Pag.)
    lower_is_better = ["% Grat. s/ Pag."]

    # Maior = melhor (mantém padrão)
    better_when_high = [
        "Pass. Transp.", "Pass. Pag.", "IPK Total", "IPK Pag.",
        "R$ Rec. p/ Veic. Conf.", "R$ Rec. p/ Pass Tot.", "R$ Rec. p/ Km rodado",
        "Pass. Tot. p/ Viagem", "Pass. Pag. p/ Viagem"
    ]

    formatters = {
        "Veículos Conf.": lambda v: fmt_float(v, 2),
        "Veículos (IDs Distintos)": fmt_int,
        "% Giro de Veículos": lambda v: fmt_pct(v, 1),
        "Km Rodada": lambda v: fmt_float(v, 1),
        "Pass. Transp.": fmt_int,
        "Pass. Grat.": fmt_int,
        "Pass. Pag.": fmt_int,
        "% Grat. s/ Pag.": lambda v: fmt_pct(v, 1),
        "IPK Total": lambda v: fmt_float(v, 3),
        "IPK Pag.": lambda v: fmt_float(v, 3),
        "R$ Rec. p/ Veic. Conf.": lambda v: fmt_currency(v, 2),
        "R$ Rec. p/ Pass Tot.": lambda v: fmt_currency(v, 2),
        "R$ Rec. p/ Km rodado": lambda v: fmt_currency(v, 2),
        "Pass. Tot. p/ Viagem": lambda v: fmt_float(v, 2),
        "Pass. Pag. p/ Viagem": lambda v: fmt_float(v, 2),
    }

    try:
        sty = tabela.style
        # Colunas maior=pior (usar colormap invertido para ficar vermelho no alto)
        cols_worse = [c for c in worse_when_high if c in tabela.columns]
        if cols_worse:
            sty = sty.background_gradient(cmap="RdYlGn_r", subset=cols_worse)
        # Colunas menor=melhor (verde no baixo) -> usar RdYlGn_r também
        cols_lower_better = [c for c in lower_is_better if c in tabela.columns]
        if cols_lower_better:
            sty = sty.background_gradient(cmap="RdYlGn_r", subset=cols_lower_better)
        # Colunas maior=melhor
        cols_better = [c for c in better_when_high if c in tabela.columns]
        if cols_better:
            sty = sty.background_gradient(cmap="RdYlGn", subset=cols_better)

        sty = sty.format(formatters)
        # Negrito na linha TOTAL
        def bold_total(row):
            return ['font-weight: bold' if row.get("Nome Linha") == "TOTAL" else '' for _ in row]
        sty = sty.apply(bold_total, axis=1)
        st.dataframe(sty, use_container_width=True)
    except Exception:
        # Fallback sem estilo, só com formatação
        for col, fn in formatters.items():
            if col in tabela.columns:
                try:
                    tabela[col] = tabela[col].apply(fn)
                except Exception:
                    pass
        st.dataframe(tabela, use_container_width=True)
else:
    st.info("Não há coluna 'Nome Linha' nos dados para consolidar.")

# ------------------------------
# Exportações
# ------------------------------
st.subheader("⬇️ Exportações")

@st.cache_data(show_spinner=False)
def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    # Usa separador ; e decimal , para PT-BR
    return df_in.to_csv(index=False, sep=";", decimal=",").encode("utf-8")

# Garante coluna de categoria no export
if "Categoria Linha" not in df_filtered.columns and "Categoria Linha" in df.columns:
    df_filtered = df_filtered.merge(df[["Nome Linha","Categoria Linha"]].drop_duplicates(), on="Nome Linha", how="left")

st.download_button("Baixar dados filtrados (CSV ;)", data=to_csv_bytes(df_filtered), file_name="dados_filtrados.csv", mime="text/csv")

# Planilha Excel com KPIs/Ranking/Financeiro/Dados
if {"Nome Linha","Passageiros"}.issubset(df_filtered.columns):
    resumo = (df_filtered.groupby("Nome Linha", as_index=False, observed=False)["Passageiros"].sum()
              .sort_values("Passageiros", ascending=False))
else:
    resumo = pd.DataFrame()

kpi_dict = {
    "Passageiros": int(total_pax) if pd.notna(total_pax) else 0,
    "Viagens": int(viagens),
    "Distância_km": float(dist_total),
    "Média_pax_por_viagem": float(media_pax),
    "Veículos_ids": int(veics_ids),
    "Linhas_ativas": int(linhas_ativas),
    "IPK_total": float(ipk_total),
    "IPK_pagantes": float(ipk_pagantes),
    "Receita_por_km": float(receita_por_km),
    "Veículos_cfg_médios": float(veic_cfg_total_medio),
    "Receita_por_veículo_cfg": float(receita_por_veic_cfg),
}
kpi_df = pd.DataFrame([kpi_dict])

@st.cache_data(show_spinner=False)
def export_xlsx(kpi_df: pd.DataFrame, resumo_df: pd.DataFrame, dados_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        kpi_df.to_excel(writer, index=False, sheet_name="KPIs")
        if not resumo_df.empty:
            resumo_df.to_excel(writer, index=False, sheet_name="Ranking_Linhas")

        # Financeiro resumido
        fin_summary = pd.DataFrame([
            {
                "Pagantes": int(total_pagantes),
                "Integrações": int(total_integracoes),
                "Gratuidades": int(total_gratuidade),
                "Tarifa_R$": float(tarifa_usuario),
                "Subsídio_por_pagante_R$": float(subsidio_pagante),
                "Receita_tarifária_R$": float(receita_tarifaria),
                "Subsídio_total_R$": float(subsidio_total),
                "Receita_total_R$": float(receita_total),
                "Custo_público_por_pax_total_R$": float(custo_publico_por_pax_total),
                "IPK_total": float(ipk_total),
                "IPK_pagantes": float(ipk_pagantes),
                "Receita_por_km": float(receita_por_km),
                "Veículos_cfg_médios": float(veic_cfg_total_medio),
                "Receita_por_veículo_cfg": float(receita_por_veic_cfg),
            }
        ])
        fin_summary.to_excel(writer, index=False, sheet_name="Financeiro")

        # Dados filtrados
        dados_df.to_excel(writer, index=False, sheet_name="Dados_Filtrados")
    return buffer.getvalue()

xlsx_bytes = export_xlsx(kpi_df, resumo, df_filtered)
st.download_button("Baixar relatório (Excel)", data=xlsx_bytes, file_name="relatorio_dashboard.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("© Dashboard ROV • Gerado automaticamente com base no arquivo e configurações persistidas.")


# ---------- Clusterização de linhas (K-Means) ----------
if ai_cluster:
    st.subheader("🧩 Clusterização de linhas (perfil operacional)")
    if not _HAS_SKLEARN:
        st.error("scikit-learn não encontrado. Instale com: `pip install scikit-learn`")
    else:
        base = df_filtered.copy()
        if "Nome Linha" not in base.columns:
            st.info("É necessário ter 'Nome Linha' para clusterização.")
        else:
            # Distância a usar
            dcol = "Distancia_cfg_km" if ("Distancia_cfg_km" in base.columns and base["Distancia_cfg_km"].notna().any()) else ("Distancia" if "Distancia" in base.columns else None)
            if dcol is None:
                base["__km__"] = np.nan
                dcol = "__km__"
            # Totais por linha
            grp = base.groupby("Nome Linha", observed=False)
            total_pax = grp["Passageiros"].sum(numeric_only=True) if "Passageiros" in base.columns else pd.Series(0, index=grp.size().index)
            total_km  = grp[dcol].sum(numeric_only=True)
            viagens   = grp.size()
            grat      = grp["Quant Gratuidade"].sum(numeric_only=True) if "Quant Gratuidade" in base.columns else pd.Series(0, index=grp.size().index)

            paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
            present_paying_c = [c for c in paying_cols_all if c in base.columns]
            if present_paying_c:
                pag_df = grp[present_paying_c].sum(numeric_only=True)
                pag = pag_df.sum(axis=1)
            else:
                pag = pd.Series(0.0, index=grp.size().index)

            receita = pag * (float(tarifa_usuario) + float(subsidio_pagante))

            # Features
            df_lin = pd.DataFrame({
                "Nome Linha": total_pax.index,
                "IPK_total": (total_pax / total_km.replace(0, np.nan)),
                "Pct_grat_s_pag": (grat / pag.replace(0, np.nan)),
                "Km_medio_viagem": (total_km / viagens.replace(0, np.nan)),
                "Rec_por_km": (receita / total_km.replace(0, np.nan)),
                "Veic_cfg_med": grp["Veiculos_cfg"].mean(numeric_only=True) if "Veiculos_cfg" in base.columns else 0.0,
                "Viagens": viagens
            }).replace([np.inf, -np.inf], np.nan)

            df_feat = df_lin.drop(columns=["Nome Linha"]).copy()
            df_feat = df_feat.fillna(df_feat.median(numeric_only=True))

            try:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(df_feat.values)
                kmeans = KMeans(n_clusters=int(k_clusters), n_init=10, random_state=42)
                labels = kmeans.fit_predict(Xs)
                df_lin["Cluster"] = labels

                # Nominar clusters por regras simples
                def nome_cluster(row):
                    ipk = row["IPK_total"]
                    kmv = row["Km_medio_viagem"]
                    rec_km = row["Rec_por_km"]
                    if pd.isna(ipk) or pd.isna(kmv) or pd.isna(rec_km):
                        return "Indefinido"
                    if ipk >= df_lin["IPK_total"].median() and kmv <= df_lin["Km_medio_viagem"].median():
                        return "Alta Demanda / Curta Distância"
                    if ipk < df_lin["IPK_total"].median() and kmv > df_lin["Km_medio_viagem"].median():
                        return "Baixa Demanda / Longa Distância"
                    if rec_km >= df_lin["Rec_por_km"].median():
                        return "Boa Receita por Km"
                    return "Misto"
                df_lin["Perfil"] = df_lin.apply(nome_cluster, axis=1)

                # Visual
                figc = px.scatter(
                    df_lin,
                    x="Km_medio_viagem", y="IPK_total",
                    color="Cluster", hover_name="Nome Linha",
                    size="Rec_por_km", title="Clusters de linhas (tamanho ~ Receita por Km)"
                )
                figc.update_layout(margin=dict(l=10,r=10,t=35,b=10), height=420)
                st.plotly_chart(figc, use_container_width=True)

                st.caption("Linhas por cluster (com métricas)")
                show_cols = ["Nome Linha","Cluster","Perfil","IPK_total","Pct_grat_s_pag","Km_medio_viagem","Rec_por_km","Veic_cfg_med","Viagens"]
                tbl = df_lin[show_cols].copy()
                # Formatação PT-BR
                tbl["IPK_total"] = tbl["IPK_total"].apply(lambda v: fmt_float(v,3))
                tbl["Pct_grat_s_pag"] = tbl["Pct_grat_s_pag"].apply(lambda v: fmt_pct(v,1))
                tbl["Km_medio_viagem"] = tbl["Km_medio_viagem"].apply(lambda v: fmt_float(v,1))
                tbl["Rec_por_km"] = tbl["Rec_por_km"].apply(lambda v: fmt_currency(v,2))
                tbl["Veic_cfg_med"] = tbl["Veic_cfg_med"].apply(lambda v: fmt_float(v,2))
                tbl["Viagens"] = tbl["Viagens"].apply(fmt_int)
                st.dataframe(tbl, use_container_width=True)

            except Exception as e:
                st.error(f"Falha na clusterização: {e}")



# === TOP KPIs (colored cards + sparklines) ====================================
import re as _re
import plotly.graph_objects as _go
import pandas as _pd
import numpy as _np
try:
    import streamlit as _st
except Exception:
    _st = None

def _kpi_fmt_int(v):
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "0"

def _kpi_fmt_float(v, dec=2):
    try:
        s = f"{float(v):,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0,00"

def _kpi_fmt_currency(v, dec=2):
    try:
        s = f"R$ {float(v):,.{dec}f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "R$ 0,00"

def _kpi_to_dt(series):
    return _pd.to_datetime(series, errors="coerce")

def _kpi_first(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _kpi_daily(df):
    d = df.copy()
    date_col = _kpi_first(d, ["Data","Data Coleta","DataColeta"]) or "Data"
    d[date_col] = _kpi_to_dt(d[date_col])
    d = d.dropna(subset=[date_col])
    d["__date"] = d[date_col].dt.date
    return d, "__date"

def _kpi_pagantes(df):
    paying = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    integ  = ["Quant Passagem Integracao","Quant Passe Integracao","Quant Vale Transporte Integracao",
              "Quant Passagem Integração","Quant Passe Integração","Quant Vale Transporte Integração"]
    regex  = [c for c in df.columns if _pd.api.types.is_numeric_dtype(df[c])
              and _re.search(r"(?i)quant.*(inteir|passag|passe|vale|vt|integra)", c)
              and not _re.search(r"(?i)grat", c)]
    cols, seen = [], set()
    for c in paying + integ + regex:
        if c in df.columns and c not in seen:
            cols.append(c); seen.add(c)
    if not cols: return _pd.Series(0, index=df.index)
    return _pd.to_numeric(df[cols], errors="coerce").fillna(0).sum(axis=1)

def _kpi_gratuitos(df):
    grat = [c for c in df.columns if _re.search(r"(?i)grat", c)]
    if not grat: return _pd.Series(0, index=df.index)
    return _pd.to_numeric(df[grat], errors="coerce").fillna(0).sum(axis=1)

def _kpi_receita_total(df):
    if "Receita Total" in df.columns:
        return _pd.to_numeric(df["Receita Total"], errors="coerce").fillna(0)
    if "Receita" in df.columns:
        return _pd.to_numeric(df["Receita"], errors="coerce").fillna(0)
    pag = _kpi_pagantes(df)
    tarifa_cols = [c for c in df.columns if _re.search(r"(?i)tarifa|valor\s*pass", c)]
    if tarifa_cols:
        tarifas = _pd.concat([_pd.to_numeric(df[c], errors="coerce") for c in tarifa_cols], axis=1).mean(axis=1).fillna(0)
    else:
        tarifas = 0
    subs_cols = [c for c in df.columns if _re.search(r"(?i)subs[ií]dio|subsidio", c)]
    subs = _pd.to_numeric(df[subs_cols[0]], errors="coerce").fillna(0) if subs_cols else 0
    return (pag * tarifas) + subs

def _kpi_dist(df):
    km_cols = [c for c in df.columns if _re.search(r"(?i)\bkm\b|quil[oô]metr", c)]
    if not km_cols: return _pd.Series(0, index=df.index)
    return _pd.to_numeric(df[km_cols[0]], errors="coerce").fillna(0)

def _kpi_compute(df):
    d, dcol = _kpi_daily(df)
    # Séries diárias
    pag_day = _kpi_pagantes(d).groupby(d[dcol]).sum(min_count=1)
    pax_day = (pag_day + _kpi_gratuitos(d).groupby(d[dcol]).sum(min_count=1))
    trips_day = d.groupby(d[dcol]).size()
    dist_day = _kpi_dist(d).groupby(d[dcol]).sum(min_count=1)
    rec_day = _kpi_receita_total(d).groupby(d[dcol]).sum(min_count=1)

    total_pax = float(pax_day.sum())
    total_trips = int(trips_day.sum())
    total_dist = float(dist_day.sum())
    rec_total = float(rec_day.sum())
    pax_per_trip = (total_pax/total_trips) if total_trips else 0.0
    ipk_pag = (float(pag_day.sum())/total_dist) if total_dist else 0.0

    veic_col = _kpi_first(df, ["Numero Veiculo","Nº Veiculo","Veiculo","Veículo"])
    linha_col = _kpi_first(df, ["Nome Linha","Linha"])
    veic_ids = df[veic_col].nunique() if veic_col else 0
    linhas_ativas = df[linha_col].nunique() if linha_col else 0

    oper_day = d.groupby(d[dcol])[veic_col].nunique() if veic_col else _pd.Series(0, index=pax_day.index)
    cfg_col = _kpi_first(df, [c for c in df.columns if "cfg" in c.lower() and "veic" in c.lower()] + ["Veiculos_cfg","Veículos_cfg"])
    if cfg_col:
        cfg_day = d.groupby(d[dcol])[cfg_col].mean()
    else:
        if linha_col and veic_col:
            per_line_daily = d.groupby([linha_col, d[dcol]])[veic_col].nunique()
            per_line_peak = per_line_daily.groupby(level=0).max()
            cap = float(per_line_peak.sum()) if not per_line_peak.empty else 0.0
        else:
            cap = float(oper_day.max() if len(oper_day) else 0.0)
        cfg_day = _pd.Series(cap, index=oper_day.index)
    ratio_day = (oper_day / cfg_day.replace(0, _np.nan)).clip(upper=1.2)

    return {
        "pax": (total_pax, pax_day),
        "trips": (total_trips, trips_day),
        "dist": (total_dist, dist_day),
        "pax_trip": (pax_per_trip, (pax_day / trips_day.replace(0,_np.nan))),
        "rec": (rec_total, rec_day),
        "ipk_pag": (ipk_pag, (pag_day / dist_day.replace(0,_np.nan))),
        "veic_ids": (veic_ids, oper_day),
        "linhas": (linhas_ativas, None),
        "ratio_op_cfg": (float((oper_day.mean() / cfg_day.mean()) if cfg_day.mean() else 0.0), ratio_day),
    }

def _kpi_delta(series, window=7, mode="window"):
    s = series.dropna() if series is not None else _pd.Series(dtype=float)
    if s.empty: return 0.0
    if mode=="previous":
        if len(s) < window*2: mode="window"
        last = float(s.iloc[-window:].mean())
        prev = float(s.iloc[-2*window:-window].mean()) if len(s) >= window*2 else float(s.mean())
    else:
        last = float(s.tail(window).mean())
        prev = float(s.mean())
    if prev == 0 or _np.isnan(prev): return 0.0
    return (last - prev) / prev

def _kpi_spark(series):
    fig = _go.Figure()
    if series is not None and not series.dropna().empty:
        s = series.dropna()
        fig.add_scatter(x=s.index, y=s.values, mode="lines")
    fig.update_layout(height=54, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

def _render_top_kpis(df):
    if _st is None: return
    series = _kpi_compute(df)

    _st.sidebar.subheader("Aparência dos KPIs")
    show_sparks = _st.sidebar.checkbox("Mostrar sparklines", True)
    compare_opt = _st.sidebar.selectbox("Comparar vs", ["Últimos 7 dias","Últimos 14 dias","Últimos 28 dias","Período anterior"], index=0)
    if compare_opt.startswith("Últimos"):
        window = int(compare_opt.split()[1]); mode="window"
    else:
        window = 7; mode="previous"

    palette = {"op":"#2563eb","fin":"#16a34a","frota":"#7c3aed","mot":"#f59e0b"}

    _st.markdown(
        """
        <style>
        .kpi-card{border-radius:16px;padding:14px 16px;background:rgba(255,255,255,0.04);
                  border:1px solid rgba(255,255,255,0.08);box-shadow:0 6px 16px rgba(0,0,0,0.24)}
        .kpi-title{font-weight:700;opacity:.9;font-size:0.95rem}
        .kpi-value{font-size:1.8rem;margin-top:4px;margin-bottom:4px}
        .kpi-delta{font-size:0.85rem;opacity:.85}
        </style>
        """,
        unsafe_allow_html=True
    )

    def card(col, title, value, color_key, series_x=None, fmt="int", suffix=""):
        with col:
            _st.markdown(f"<div class='kpi-title'>{title}</div>", unsafe_allow_html=True)
            if fmt=="int": val_fmt = _kpi_fmt_int(value)
            elif fmt=="float": val_fmt = _kpi_fmt_float(value, 2)
            elif fmt=="currency": val_fmt = _kpi_fmt_currency(value, 2)
            else: val_fmt = str(value)
            _st.markdown(f"<div class='kpi-value' style='color:{palette[color_key]}'>{val_fmt}{suffix}</div>", unsafe_allow_html=True)
            if series_x is not None:
                delta = _kpi_delta(series_x, window, mode)
                badge = "🟢" if delta >= 0.02 else ("🟡" if delta > -0.02 else "🔴")
                _st.markdown(f"<div class='kpi-delta'>{badge} Δ {delta*100:.1f}%</div>", unsafe_allow_html=True)
            if show_sparks and series_x is not None:
                _st.plotly_chart(_kpi_spark(series_x), use_container_width=True, config={"displayModeBar": False})

    # Row 1
    r1 = _st.columns(3)
    card(r1[0], "🧍 Passageiros", series["pax"][0], "op", series["pax"][1], "int")
    card(r1[1], "🧾 Viagens registradas", series["trips"][0], "op", series["trips"][1], "int")
    card(r1[2], "👥 Média pax/viagem", series["pax_trip"][0], "op", series["pax_trip"][1], "float")

    # Row 2
    r2 = _st.columns(3)
    card(r2[0], "💰 Receita total", series["rec"][0], "fin", series["rec"][1], "currency")
    card(r2[1], "📈 IPK pagantes (pax/km)", series["ipk_pag"][0], "fin", series["ipk_pag"][1], "float")
    ratio_val, ratio_series = series["ratio_op_cfg"]
    with r2[2]:
        pct = f"{max(0.0, min(1.2, float(ratio_val)))*100:.1f}%"
        badge = "🟢" if ratio_val >= 0.9 else ("🟡" if ratio_val >= 0.75 else "🔴")
        _st.markdown(f"<div class='kpi-title'>🚌 Operação vs Config</div>", unsafe_allow_html=True)
        _st.markdown(f"<div class='kpi-value' style='color:{palette['frota']}'>{badge} {pct}</div>", unsafe_allow_html=True)
        if show_sparks and ratio_series is not None:
            _st.plotly_chart(_kpi_spark(ratio_series), use_container_width=True, config={"displayModeBar": False})
# === END TOP KPIs =============================================================


# ========= HOTFIX: normalização robusta de datas =========
def _ensure_datetime_columns(df):
    try:
        date_cols = [
            "Data", "Data Coleta", "DataColeta",
            "Data Hora Inicio Operacao", "Data Hora Início Operação", "Inicio Operacao", "Início Operação", "Hora Inicio", "DataHoraInicio",
            "Data Hora Final Operacao", "Data Hora Final Operação", "Fim Operacao", "Hora Final", "DataHoraFim"
        ]
        for c in date_cols:
            if c in df.columns:
                try:
                    if 'is_dt64' in globals():
                        if not is_dt64(df[c]):
                            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                    else:
                        df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                except Exception:
                    df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
    except Exception:
        pass
    return df

try:
    if 'df' in globals() and isinstance(df, pd.DataFrame):
        df = _ensure_datetime_columns(df)
    if 'df_filtered' in globals() and isinstance(df_filtered, pd.DataFrame):
        df_filtered = _ensure_datetime_columns(df_filtered)
except Exception:
    pass
# ========= FIM HOTFIX =========
