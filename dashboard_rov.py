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
    # Motoristas distintos (robusto ao período filtrado)
    # Preferir 'Matricula'; se ausente, normalizar nome do motorista
    try:
        import unicodedata as _ud
        def _norm_drv(x):
            if pd.isna(x):
                return None
            s = str(x).strip()
            if not s:
                return None
            s = ''.join(c for c in _ud.normalize("NFKD", s) if not _ud.combining(c))
            s = ' '.join(s.split()).upper()
            return s
        if "Matricula" in df_filtered.columns and df_filtered["Matricula"].notna().any():
            _drv = df_filtered["Matricula"].astype(str).str.strip()
            _invalid = _drv.eq("") | _drv.str.lower().isin(["nan","none","null","-"])
            n_motoristas = int(_drv[~_invalid].nunique())
        elif "Cobrador/Operador" in df_filtered.columns and df_filtered["Cobrador/Operador"].notna().any():
            _drv = df_filtered["Cobrador/Operador"].map(_norm_drv)
            _invalid = _drv.isna() | _drv.isin({"TREINAMENTO","TESTE","SEM MOTORISTA","S/MOTORISTA","S MOTORISTA","N/A","NA"})
            n_motoristas = int(_drv[~_invalid].nunique())
        else:
            n_motoristas = 0
    except Exception:
        # fallback para lógica anterior caso algo dê errado
        n_motoristas = len(viagens_m.index)

    k1, k2, k3, k4, k5 = st.columns(5)
# === Cálculo robusto de Motoristas distintos (no período filtrado) ===
# Preferimos 'Matricula' (identificador estável). Se ausente, usamos nome normalizado.
import unicodedata as _ud
import re as _re

def _norm_text_driver(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    s = ''.join(c for c in _ud.normalize("NFKD", s) if not _ud.combining(c))
    s = _re.sub(r'\s+', ' ', s).upper()
    return s

n_motoristas = 0
if "Matricula" in df_filtered.columns and df_filtered["Matricula"].notna().any():
    drv_series = df_filtered["Matricula"].astype(str).str.strip()
    invalid = drv_series.eq("") | drv_series.str.lower().isin(["nan","none","null","-"])
    n_motoristas = int(drv_series[~invalid].nunique())
elif "Cobrador/Operador" in df_filtered.columns and df_filtered["Cobrador/Operador"].notna().any():
    drv_series = df_filtered["Cobrador/Operador"].map(_norm_text_driver)
    invalid = drv_series.isna() | drv_series.isin({"TREINAMENTO","TESTE","SEM MOTORISTA","S/MOTORISTA","S MOTORISTA","N/A","NA"})
    n_motoristas = int(drv_series[~invalid].nunique())
else:
    n_motoristas = 0
# === fim cálculo robusto motoristas ===
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

    # Colunas de pagantes (inclui integrações) + fallback por regex
    paying_cols_all = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte"]
    integration_cols_all = ["Quant Passagem Integracao","Quant Passe Integracao","Quant Vale Transporte Integracao",
                            "Quant Passagem Integração","Quant Passe Integração","Quant Vale Transporte Integração"]
    present_paying_l = [c for c in paying_cols_all if c in base_tbl.columns]
    present_integration_l = [c for c in integration_cols_all if c in base_tbl.columns]
    # Fallback: qualquer coluna numérica com "Quant" e termos de pagantes (exclui gratuidade)
    regex_candidates = [c for c in base_tbl.columns if re.search(r"(?i)quant.*(inteir|passag|passe|vale|vt|integra)", c) and not re.search(r"(?i)grat", c)]
    candidate_cols = []
    # mantém ordem e remove duplicados
    for c in present_paying_l + present_integration_l + regex_candidates:
        if c not in candidate_cols:
            candidate_cols.append(c)
    if candidate_cols:
        pag_by_cols_tbl = grp[candidate_cols].sum(numeric_only=True)
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
        "R$ Receita Total": receita_tot_l_tbl,
        "Tot. Viagens": viagens_total_tbl,
        "Viagens p/ Veic": (viagens_total_tbl / veic_ids_uni_tbl.replace(0, np.nan)),
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
        "R$ Receita Total": total_receita_sum,
        "Tot. Viagens": total_viagens_sum,
        "Viagens p/ Veic": (total_viagens_sum / total_veic_ids_sum) if total_veic_ids_sum else np.nan,
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
    , "R$ Receita Total", "Tot. Viagens", "Viagens p/ Veic"]

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
        "R$ Receita Total": lambda v: fmt_currency(v, 2),
        "Tot. Viagens": lambda v: fmt_int(v),
        "Viagens p/ Veic": lambda v: fmt_float(v, 2),
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

# ===================================================================
# ### 🚚 Aproveitamento da Frota (KPIs + por linha)
# ===================================================================
import pandas as _pd
import numpy as _np
try:
    import streamlit as _st
except Exception:
    _st = None

def _first_present(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def _to_dt(series):
    return _pd.to_datetime(series, errors="coerce")


def _calc_aproveitamento(df):
    """
    Calcula KPIs gerais e tabela por linha de aproveitamento da frota.
    ATENÇÃO: KPIs agora são calculados em nível de SISTEMA por dia,
    evitando médias de médias que distorcem valores.
    Retorna (kpis_dict, tabela_por_linha DataFrame).
    """
    import pandas as _pd
    import numpy as _np

    if df is None or df.empty:
        return {}, _pd.DataFrame()

    col_linha = _first_present(df, ["Nome Linha","Linha"])
    col_veic  = _first_present(df, ["Numero Veiculo","Nº Veiculo","Veiculo","Veículo"])
    col_data  = _first_present(df, ["Data","Data Coleta","DataColeta"])
    col_ini   = _first_present(df, ["Data Hora Inicio Operacao","Data Hora Início Operação","Inicio Operacao","Início Operação","Hora Inicio","DataHoraInicio"])
    col_fim   = _first_present(df, ["Data Hora Final Operacao","Data Hora Final Operação","Fim Operacao","Hora Final","DataHoraFim"])

    if any(c is None for c in [col_linha, col_veic, col_data, col_ini, col_fim]):
        return {"erro":"Colunas de linha/veículo/horários ausentes"}, _pd.DataFrame()

    dff = df.copy()
    dff[col_data] = _to_dt(dff[col_data])
    dff[col_ini]  = _to_dt(dff[col_ini])
    dff[col_fim]  = _to_dt(dff[col_fim])
    dff["_horas"] = ((dff[col_fim] - dff[col_ini]).dt.total_seconds() / 3600.0).fillna(0)

    # ----- Séries diárias (nível sistema) -----
    # Horas totais por dia (somando todas as viagens/veículos)
    daily_hours = dff.groupby(col_data)["_horas"].sum(min_count=1)
    dias_ativos = int(daily_hours.index.nunique())

    # Veículos operando por dia (distintos)
    daily_oper = dff.groupby(col_data)[col_veic].nunique()

    # Veículos configurados por dia:
    #  - Se houver coluna de config, somamos a config por linha ao dia e tiramos média entre dias
    #  - Senão, aproximamos pela "capacidade" do sistema como soma do pico (máximo diário) por linha
    veic_cfg_col = _first_present(dff, [c for c in dff.columns if "cfg" in c.lower() and "veic" in c.lower()] + ["Veiculos_cfg","Veículos_cfg"])

    if veic_cfg_col:
        daily_cfg = dff.groupby([col_data, col_linha])[veic_cfg_col].mean().groupby(level=0).sum(min_count=1)
        veic_cfg_med = float(daily_cfg.mean()) if not daily_cfg.empty else 0.0
    else:
        per_line_daily = dff.groupby([col_linha, col_data])[col_veic].nunique()
        per_line_peak  = per_line_daily.groupby(level=0).max()  # pico por linha no período
        cfg_total_cap  = float(per_line_peak.sum()) if not per_line_peak.empty else 0.0
        veic_cfg_med   = cfg_total_cap  # aproxima como capacidade estável
        daily_cfg      = _pd.Series(cfg_total_cap, index=daily_hours.index) if dias_ativos else _pd.Series(dtype=float)

    # KPIs em nível de sistema (médias por dia)
    horas_totais     = float(daily_hours.sum())                         # total no período
    horas_dia_media  = float(daily_hours.mean()) if dias_ativos else 0  # média diária
    veic_med_op      = float(daily_oper.mean()) if dias_ativos else 0
    horas_por_cfg    = (horas_dia_media / veic_cfg_med) if veic_cfg_med else 0.0
    horas_por_oper   = (horas_dia_media / veic_med_op) if veic_med_op else 0.0
    ratio_oper_cfg   = (veic_med_op / veic_cfg_med) if veic_cfg_med else 0.0
    # Limita ratio em 120% para evitar outliers visuais por ruído de dados
    ratio_oper_cfg   = float(min(ratio_oper_cfg, 1.2))

    kpis = {
        "horas_totais": horas_totais,
        "dias_ativos": dias_ativos,
        "veic_med_op": veic_med_op,
        "veic_cfg_med": veic_cfg_med,
        "horas_dia_media": horas_dia_media,
        "horas_por_cfg": horas_por_cfg,
        "horas_por_oper": horas_por_oper,
        "ratio_oper_cfg": ratio_oper_cfg,
    }

    # ---- Tabela por linha (mantém lógica robusta) ----
    grp = dff.groupby(col_linha, dropna=False)
    horas_totais_l = grp["_horas"].sum().rename("Horas totais")
    dias_ativos_l  = grp[col_data].nunique().rename("Dias ativos")

    def _vm(g):
        return g.groupby(col_data)[col_veic].nunique().mean()
    veic_med_op_l = dff.groupby(col_linha).apply(_vm).rename("Veic. médios operação/dia")

    if veic_cfg_col:
        veic_cfg_med_l = dff.groupby([col_linha, col_data])[veic_cfg_col].mean().groupby(level=0).mean().rename("Veic. configurados (média)")
    else:
        veic_cfg_med_l = dff.groupby(col_linha).apply(lambda g: g.groupby(col_data)[col_veic].nunique().max()).rename("Veic. configurados (média)")

    base = _pd.concat([horas_totais_l, dias_ativos_l, veic_med_op_l, veic_cfg_med_l], axis=1).reset_index()
    base["Horas/dia (média)"] = base.apply(lambda r: (r["Horas totais"]/r["Dias ativos"]) if r["Dias ativos"] else 0, axis=1)
    base["Horas/dia por veic. cfg"] = base.apply(lambda r: (r["Horas/dia (média)"]/r["Veic. configurados (média)"]) if r["Veic. configurados (média)"] else 0, axis=1)
    base["Horas/dia por veic. oper. méd."] = base.apply(lambda r: (r["Horas/dia (média)"]/r["Veic. médios operação/dia"]) if r["Veic. médios operação/dia"] else 0, axis=1)
    base["Operação vs Config (ratio)"] = base.apply(lambda r: (r["Veic. médios operação/dia"]/r["Veic. configurados (média)"]) if r["Veic. configurados (média)"] else 0, axis=1)

    base = base.sort_values(["Operação vs Config (ratio)", "Horas/dia (média)"], ascending=False)
    return kpis, base

# ---- Renderização na UI ----
try:
    _base_df = df_filtered.copy() if 'df_filtered' in globals() else df.copy()

    # Helpers de formatação
    def _fmt_h(v, dec=1):
        try:
            return f"{v:,.{dec}f} h".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            return "—"

    def _fmt_hhmm(v):
        try:
            total_min = int(round(float(v) * 60))
            hh, mm = divmod(total_min, 60)
            return f"{hh:02d}:{mm:02d} h"
        except Exception:
            return "—"

    def _fmt_pct(v):
        try:
            return f"{(float(v)*100):.1f}%"
        except Exception:
            return "—"

    if _st: _st.markdown("## 🚚 Aproveitamento da Frota")
    _kpis, _tbl = _calc_aproveitamento(_base_df)

    if _kpis.get("erro") and _st:
        _st.info("Não foi possível calcular: " + _kpis["erro"])
    elif _st:
        c1,c2,c3 = _st.columns(3)
        c1.metric("⏱️ Horas totais", _fmt_h(_kpis['horas_totais'], 1), _fmt_hhmm(_kpis['horas_dia_media']))
        c2.metric("🗓️ Dias ativos", f"{_kpis['dias_ativos']}")
        c3.metric("🚌 Veic. médios oper./dia", f"{_kpis['veic_med_op']:.1f}")

        c4,c5,c6 = _st.columns(3)
        c4.metric("🚐 Veic. configurados (média)", f"{_kpis['veic_cfg_med']:.1f}")
        c5.metric("⏳ Horas/dia por veic. cfg", f"{_kpis['horas_por_cfg']:.2f}")
        # Badge de status para Operação vs Config
        ratio = _kpis['ratio_oper_cfg']
        badge = "🟢" if ratio >= 0.9 else ("🟡" if ratio >= 0.75 else "🔴")
        c6.metric(f"{badge} Operação vs Config", _fmt_pct(ratio))

        _st.caption("• Delta em 'Horas totais' mostra média diária (formato HH:MM). Cores: 🟢 ≥ 90%, 🟡 75–89%, 🔴 < 75%.")

        _st.markdown("### Por linha")
        tbl_show = _tbl.copy()
        if not tbl_show.empty and "Operação vs Config (ratio)" in tbl_show.columns:
            tbl_show["Operação vs Config (ratio)"] = (tbl_show["Operação vs Config (ratio)"].astype(float).clip(0,1.2))*100.0
            tbl_show["Operação vs Config (ratio)"] = tbl_show["Operação vs Config (ratio)"].map(lambda v: f"{v:.1f}%")
        # Formata horas
        for col in ["Horas totais","Horas/dia (média)","Horas/dia por veic. cfg","Horas/dia por veic. oper. méd."]:
            if col in tbl_show.columns:
                tbl_show[col] = tbl_show[col].astype(float).map(lambda v: _fmt_h(v, 2))
        _st.dataframe(tbl_show, use_container_width=True)

except Exception as _e:
    if _st: _st.warning(f"Falha ao renderizar 'Aproveitamento da Frota': {_e}")







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
