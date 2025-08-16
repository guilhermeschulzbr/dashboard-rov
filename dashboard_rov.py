import streamlit as st
import pandas as pd

# ------------------------------
# Configuração da página
# ------------------------------
st.set_page_config(page_title="Dashboard ROV - Operação",
                   layout="wide",
                   initial_sidebar_state="expanded")

# ==============================
# Utilidades comuns
# ==============================
def _fmt_hhmm(total_min):
    try:
        total_min = int(round(float(total_min)))
    except Exception:
        total_min = 0
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"

def _detect_motorista_col(df):
    M_CANDS = ["Motorista","Operador","Cobrador/Operador","MOTORISTA","Matricula","Matrícula",
               "CPF Motorista","ID Motorista","Nome Motorista","Nome do Motorista"]
    return next((c for c in M_CANDS if c in df.columns), None)

# ==============================
# Painel: Linha do tempo Veículo × Linha (1 dia)
# ==============================
def show_linha_do_tempo_alocacao_1dia(df, titulo="📆 Linha do tempo de alocação (1 dia)"):
    import plotly.express as px
    vcol, lcol = "Numero Veiculo", "Nome Linha"
    scol, ecol = "Data Hora Inicio Operacao", "Data Hora Final Operacao"

    miss = [c for c in [vcol,lcol,scol,ecol] if c not in df.columns]
    if miss:
        st.error("Colunas ausentes: " + ", ".join(miss)); return

    # Total de passageiros (pagantes + gratuidades, se campo total não existir)
    CAND_PASS_TOTAL = ["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros",
                       "Total Passageiros","Passageiros Transportados","Qtd de Passageiros",
                       "Quantidade de Passageiros"]
    CAND_PAGANTES   = ["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte",
                       "Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes",
                       "Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT       = ["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade",
                       "Gratuidades","Gratuidade","Quantidade Gratuidade"]

    def _num_from_row(row, cols):
        tot = 0.0
        for c in cols:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v): tot += float(v)
        return tot

    def _pax_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v = pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        return _num_from_row(row, CAND_PAGANTES) + _num_from_row(row, CAND_GRAT)

    st.markdown("## " + titulo)
    df = df.copy()
    df[scol] = pd.to_datetime(df[scol], errors="coerce")
    df[ecol] = pd.to_datetime(df[ecol], errors="coerce")

    sdates, edates = df[scol].dropna().dt.date, df[ecol].dropna().dt.date
    from datetime import date as _date
    day_default = sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())
    dia = st.date_input("Dia (1 dia)", value=day_default, format="DD/MM/YYYY", key="base_dia")
    day_start = pd.Timestamp(dia).normalize(); day_end = day_start + pd.Timedelta(days=1)

    pass_cols = [c for c in (CAND_PASS_TOTAL+CAND_PAGANTES+CAND_GRAT) if c in df.columns]
    tmp = df[[vcol,lcol,scol,ecol]+pass_cols].dropna(subset=[vcol,scol,ecol]).copy()

    segs = []
    for _,r in tmp.iterrows():
        s,e = r[scol], r[ecol]
        if pd.isna(s) or pd.isna(e) or e<=day_start or s>=day_end: continue
        s2,e2 = max(s,day_start), min(e,day_end)
        if s2>=e2: continue
        segs.append({"Veículo": str(r[vcol]), "Linha": str(r[lcol]), "Início": s2, "Fim": e2, "Passageiros": _pax_row(r)})
    seg = pd.DataFrame(segs).sort_values(["Veículo","Início","Fim"])
    if seg.empty: st.info("Sem segmentos para o dia."); return

    # completar ociosos por veículo
    ociosos=[]
    for veic,g in seg.groupby("Veículo", sort=False):
        cur=day_start
        for _,rr in g.iterrows():
            if rr["Início"]>cur: ociosos.append({"Veículo":veic,"Linha":"Ocioso","Início":cur,"Fim":rr["Início"],"Passageiros":0.0})
            cur=max(cur, rr["Fim"])
        if cur<day_end: ociosos.append({"Veículo":veic,"Linha":"Ocioso","Início":cur,"Fim":day_end,"Passageiros":0.0})
    if ociosos:
        seg = pd.concat([seg, pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Veículo","Início"])

    seg["Duração (min)"] = (seg["Fim"]-seg["Início"]).dt.total_seconds()/60.0

    with st.expander("Filtros de exibição"):
        veics = sorted(seg["Veículo"].astype(str).unique().tolist())
        linhas = sorted(seg["Linha"].astype(str).unique().tolist())
        pick_veics = st.multiselect("Filtrar Veículos", veics, default=veics, key="base_filt_veic")
        pick_linhas = st.multiselect("Filtrar Linhas (inclui 'Ocioso')", linhas, default=linhas, key="base_filt_lin")
        segf = seg[ seg["Veículo"].isin(pick_veics) & seg["Linha"].astype(str).isin(pick_linhas) ]
    if segf.empty: st.info("Filtros vazios."); return

    # Indicadores
    segf = segf.copy(); segf["_dur_min"] = (segf["Fim"]-segf["Início"]).dt.total_seconds()/60.0
    work = segf["Linha"].astype(str)!="Ocioso"
    t_work = float(segf.loc[work,"_dur_min"].sum()); t_idle=float(segf.loc[~work,"_dur_min"].sum()); t_tot=t_work+t_idle
    pax_tot = float(pd.to_numeric(segf.loc[work,"Passageiros"], errors="coerce").fillna(0).sum())
    pph = (pax_tot/(t_work/60.0)) if t_work>0 else 0.0
    _pct=lambda x:f"{(float(x)*100):.1f}%"
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Veículos", f"{segf['Veículo'].nunique()}"); c2.metric("Tempo Operacional", _fmt_hhmm(t_work))
    c3.metric("Tempo Ocioso", _fmt_hhmm(t_idle)); c4.metric("% Ocioso", _pct(t_idle/t_tot) if t_tot>0 else "0.0%")
    c5.metric("Passageiros/Hora", f"{pph:.1f}")

    # Flag ZeroPass para padronizar hover/pattern (não-ocioso)
    segf["ZeroPass"] = (segf["Linha"].astype(str)!="Ocioso") & (pd.to_numeric(segf["Passageiros"], errors="coerce").fillna(-1)==0)

    fig = px.timeline(segf, x_start="Início", x_end="Fim", y="Veículo", color="Linha",
                      pattern_shape="ZeroPass", pattern_shape_map={True:"x", False:""},
                      hover_data={"Duração (min)":":.1f","Passageiros":True,"ZeroPass":True,"Veículo":True,"Linha":True,"Início":True,"Fim":True})
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Veículo")
    st.plotly_chart(fig, use_container_width=True)

    # Dados + CSV
    st.markdown("#### Dados — Veículo × Linha (1 dia)")
    if not segf.empty:
        st.dataframe(segf, use_container_width=True, hide_index=True)
        _fname = f"alocacao_veiculos_linhas_{pd.to_datetime(dia).strftime('%Y%m%d')}.csv"
        _csv = segf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV – Veículo × Linha (1 dia)", data=_csv, file_name=_fname, mime="text/csv", key="dl_aloc_base")
    else:
        st.caption("Sem dados para exportar neste momento.")

# ==============================
# Painel: Linha do tempo — Motoristas × Linhas (1 dia)
# ==============================
def show_linha_do_tempo_motoristas_linhas_1dia(df, titulo="📆 Linha do tempo: Motoristas × Linhas (1 dia)"):
    import plotly.express as px
    lcol="Nome Linha"; scol="Data Hora Inicio Operacao"; ecol="Data Hora Final Operacao"
    mcol=_detect_motorista_col(df)
    miss=[c for c in [mcol,lcol,scol,ecol] if c is None or c not in df.columns]
    if miss: st.error("Colunas ausentes (Mot × Linhas): " + ", ".join(map(str,miss))); return

    st.markdown("## " + titulo)
    df=df.copy(); df[scol]=pd.to_datetime(df[scol], errors="coerce"); df[ecol]=pd.to_datetime(df[ecol], errors="coerce")
    sdates,edates=df[scol].dropna().dt.date, df[ecol].dropna().dt.date
    from datetime import date as _date
    day_default=sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())
    dia=st.date_input("Dia (1 dia) — Motoristas × Linhas", value=day_default, format="DD/MM/YYYY", key="ml_dia")
    day_start=pd.Timestamp(dia).normalize(); day_end=day_start+pd.Timedelta(days=1)

    CAND_PASS_TOTAL=["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros","Total Passageiros","Passageiros Transportados","Qtd de Passageiros","Quantidade de Passageiros"]
    CAND_PAGANTES=["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte","Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes","Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT=["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade","Gratuidades","Gratuidade","Quantidade Gratuidade"]
    def _num_row(row, cols):
        tot=0.0
        for c in cols:
            if c in row.index:
                v=pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v): tot+=float(v)
        return tot
    def _pax_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v=pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        return _num_row(row,CAND_PAGANTES)+_num_row(row,CAND_GRAT)

    pass_cols=[c for c in (CAND_PASS_TOTAL+CAND_PAGANTES+CAND_GRAT) if c in df.columns]
    tmp=df[[mcol,lcol,scol,ecol]+pass_cols].dropna(subset=[mcol,scol,ecol]).copy()
    segs=[]
    for _,r in tmp.iterrows():
        s,e=r[scol],r[ecol]
        if pd.isna(s) or pd.isna(e) or e<=day_start or s>=day_end: continue
        s2,e2=max(s,day_start),min(e,day_end)
        if s2>=e2: continue
        segs.append({"Motorista":str(r[mcol]),"Linha":str(r[lcol]),"Início":s2,"Fim":e2,"Passageiros":_pax_row(r)})
    seg=pd.DataFrame(segs)
    if seg.empty: st.info("Sem segmentos."); return
    seg=seg.sort_values(["Motorista","Início","Fim"])
    ociosos=[]
    for mot,g in seg.groupby("Motorista", sort=False):
        cur=day_start
        for _,rr in g.iterrows():
            if rr["Início"]>cur: ociosos.append({"Motorista":mot,"Linha":"Ocioso","Início":cur,"Fim":rr["Início"],"Passageiros":0.0})
            cur=max(cur, rr["Fim"])
        if cur<day_end: ociosos.append({"Motorista":mot,"Linha":"Ocioso","Início":cur,"Fim":day_end,"Passageiros":0.0})
    if ociosos: seg=pd.concat([seg,pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Motorista","Início"])
    seg["Duração (min)"]=(seg["Fim"]-seg["Início"]).dt.total_seconds()/60.0

    with st.expander("Filtros — Motoristas × Linhas"):
        mot_list=sorted(seg["Motorista"].astype(str).unique().tolist())
        linhas=sorted(seg["Linha"].astype(str).unique().tolist())
        pick_mot=st.multiselect("Filtrar Motoristas", mot_list, default=mot_list, key="ml_filt_mot")
        pick_lin=st.multiselect("Filtrar Linhas (inclui 'Ocioso')", linhas, default=linhas, key="ml_filt_lin")
        segf=seg[ seg["Motorista"].isin(pick_mot) & seg["Linha"].astype(str).isin(pick_lin) ]
    if segf.empty: st.info("Filtros vazios."); return

    # Totais + rótulo Motorista_Label com HE
    segf=segf.copy(); segf["_dur_min"]=(segf["Fim"]-segf["Início"]).dt.total_seconds()/60.0
    work_mask=segf["Linha"].astype(str)!="Ocioso"
    work_per_mot=segf.loc[work_mask].groupby("Motorista", observed=False)["_dur_min"].sum()
    limite=7*60+20
    mot_label_map={}
    for mot,mins in work_per_mot.items():
        extra=max(0, mins-limite)
        if extra>0:
            mot_label_map[mot]=f"⚡ <b>{mot} — {_fmt_hhmm(mins)} (HE {_fmt_hhmm(extra)})</b>"
        else:
            mot_label_map[mot]=f"{mot} — {_fmt_hhmm(mins)}"
    segf["Motorista_Label"]=segf["Motorista"].map(mot_label_map).fillna(segf["Motorista"].astype(str))

    segf["ZeroPass"]=(segf["Linha"].astype(str)!="Ocioso") & (pd.to_numeric(segf["Passageiros"], errors="coerce").fillna(-1)==0)

    fig=px.timeline(segf, x_start="Início", x_end="Fim", y="Motorista_Label", color="Linha",
                    pattern_shape="ZeroPass", pattern_shape_map={True:"x",False:""},
                    hover_data={"Duração (min)":":.1f","Passageiros":True,"ZeroPass":True,"Motorista":True,"Linha":True,"Início":True,"Fim":True})
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Motorista")
    st.plotly_chart(fig, use_container_width=True)

    # Dados + CSV
    st.markdown("#### Dados — Motoristas × Linhas (1 dia)")
    if not segf.empty:
        st.dataframe(segf, use_container_width=True, hide_index=True)
        _fname = f"motoristas_x_linhas_{pd.to_datetime(dia).strftime('%Y%m%d')}.csv"
        _csv = segf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV – Motoristas × Linhas (1 dia)", data=_csv, file_name=_fname, mime="text/csv", key="dl_ml_1d")
    else:
        st.caption("Sem dados para exportar neste momento.")

# ==============================
# Painel: Linha do tempo — Motoristas × Veículos (1 dia)
# ==============================
def show_linha_do_tempo_motoristas_veiculos_1dia(df, titulo="📆 Linha do tempo: Motoristas × Veículos (1 dia)"):
    import plotly.express as px
    vcol="Numero Veiculo"; scol="Data Hora Inicio Operacao"; ecol="Data Hora Final Operacao"
    mcol=_detect_motorista_col(df)
    miss=[c for c in [mcol,vcol,scol,ecol] if c is None or c not in df.columns]
    if miss: st.error("Colunas ausentes (Mot × Veículos): " + ", ".join(map(str,miss))); return

    st.markdown("## " + titulo)
    df=df.copy(); df[scol]=pd.to_datetime(df[scol], errors="coerce"); df[ecol]=pd.to_datetime(df[ecol], errors="coerce")
    sdates,edates=df[scol].dropna().dt.date, df[ecol].dropna().dt.date
    from datetime import date as _date
    day_default=sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today())
    dia=st.date_input("Dia (1 dia) — Motoristas × Veículos", value=day_default, format="DD/MM/YYYY", key="mv_dia")
    day_start=pd.Timestamp(dia).normalize(); day_end=day_start+pd.Timedelta(days=1)

    CAND_PASS_TOTAL=["Passageiros","Qtd Passageiros","Qtde Passageiros","Quantidade Passageiros","Total Passageiros","Passageiros Transportados","Qtd de Passageiros","Quantidade de Passageiros"]
    CAND_PAGANTES=["Quant Inteiras","Quant Passagem","Quant Passe","Quant Vale Transporte","Pagantes","Quantidade Pagantes","Qtd Pagantes","Qtde Pagantes","Validações","Validacoes","Validacao","Validação","Embarques","Embarcados"]
    CAND_GRAT=["Quant Gratuidade","Qtd Gratuidade","Qtde Gratuidade","Gratuidades","Gratuidade","Quantidade Gratuidade"]
    def _num_row(row, cols):
        tot=0.0
        for c in cols:
            if c in row.index:
                v=pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v): tot+=float(v)
        return tot
    def _pax_row(row):
        for c in CAND_PASS_TOTAL:
            if c in row.index:
                v=pd.to_numeric(row[c], errors="coerce")
                return 0.0 if pd.isna(v) else float(v)
        return _num_row(row,CAND_PAGANTES)+_num_row(row,CAND_GRAT)

    pass_cols=[c for c in (CAND_PASS_TOTAL+CAND_PAGANTES+CAND_GRAT) if c in df.columns]
    tmp=df[[mcol,vcol,scol,ecol]+pass_cols].dropna(subset=[mcol,vcol,scol,ecol]).copy()
    segs=[]
    for _,r in tmp.iterrows():
        s,e=r[scol],r[ecol]
        if pd.isna(s) or pd.isna(e) or e<=day_start or s>=day_end: continue
        s2,e2=max(s,day_start),min(e,day_end)
        if s2>=e2: continue
        segs.append({"Motorista":str(r[mcol]),"Veículo":str(r[vcol]),"Início":s2,"Fim":e2,"Passageiros":_pax_row(r)})
    seg=pd.DataFrame(segs)
    if seg.empty: st.info("Sem segmentos."); return
    seg=seg.sort_values(["Motorista","Início","Fim"])
    ociosos=[]
    for mot,g in seg.groupby("Motorista", sort=False):
        cur=day_start
        for _,rr in g.iterrows():
            if rr["Início"]>cur: ociosos.append({"Motorista":mot,"Veículo":"Ocioso","Início":cur,"Fim":rr["Início"],"Passageiros":0.0})
            cur=max(cur, rr["Fim"])
        if cur<day_end: ociosos.append({"Motorista":mot,"Veículo":"Ocioso","Início":cur,"Fim":day_end,"Passageiros":0.0})
    if ociosos: seg=pd.concat([seg,pd.DataFrame(ociosos)], ignore_index=True).sort_values(["Motorista","Início"])
    seg["Duração (min)"]=(seg["Fim"]-seg["Início"]).dt.total_seconds()/60.0

    with st.expander("Filtros — Motoristas × Veículos"):
        mot_list=sorted(seg["Motorista"].astype(str).unique().tolist())
        veics=sorted(seg["Veículo"].astype(str).unique().tolist())
        pick_mot=st.multiselect("Filtrar Motoristas", mot_list, default=mot_list, key="mv_filt_mot")
        pick_vei=st.multiselect("Filtrar Veículos (inclui 'Ocioso')", veics, default=veics, key="mv_filt_vei")
        segf=seg[ seg["Motorista"].isin(pick_mot) & seg["Veículo"].astype(str).isin(pick_vei) ]
    if segf.empty: st.info("Filtros vazios."); return

    # Totais + rótulo Motorista_Label com HE
    segf=segf.copy(); segf["_dur_min"]=(segf["Fim"]-segf["Início"]).dt.total_seconds()/60.0
    work_mask=segf["Veículo"].astype(str)!="Ocioso"
    work_per_mot=segf.loc[work_mask].groupby("Motorista", observed=False)["_dur_min"].sum()
    limite=7*60+20
    mot_label_map={}
    for mot,mins in work_per_mot.items():
        extra=max(0, mins-limite)
        if extra>0:
            mot_label_map[mot]=f"⚡ <b>{mot} — {_fmt_hhmm(mins)} (HE {_fmt_hhmm(extra)})</b>"
        else:
            mot_label_map[mot]=f"{mot} — {_fmt_hhmm(mins)}"
    segf["Motorista_Label"]=segf["Motorista"].map(mot_label_map).fillna(segf["Motorista"].astype(str))

    segf["ZeroPass"]=(segf["Veículo"].astype(str)!="Ocioso") & (pd.to_numeric(segf["Passageiros"], errors="coerce").fillna(-1)==0)

    import plotly.express as px
    fig=px.timeline(segf, x_start="Início", x_end="Fim", y="Motorista_Label", color="Veículo",
                    pattern_shape="ZeroPass", pattern_shape_map={True:"x",False:""},
                    hover_data={"Duração (min)":":.1f","Passageiros":True,"ZeroPass":True,"Motorista":True,"Veículo":True,"Início":True,"Fim":True})
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(height=650, xaxis_title="Horário", yaxis_title="Motorista")
    st.plotly_chart(fig, use_container_width=True)

    # Dados + CSV
    st.markdown("#### Dados — Motoristas × Veículos (1 dia)")
    if not segf.empty:
        st.dataframe(segf, use_container_width=True, hide_index=True)
        _fname = f"motoristas_x_veiculos_{pd.to_datetime(dia).strftime('%Y%m%d')}.csv"
        _csv = segf.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV – Motoristas × Veículos (1 dia)", data=_csv, file_name=_fname, mime="text/csv", key="dl_mv_1d")
    else:
        st.caption("Sem dados para exportar neste momento.")

# ==============================
# Painel: Rotatividade Motoristas × Veículos (período)
# ==============================
def show_rotatividade_motoristas_veiculos(df, titulo="🔁 Rotatividade Motoristas × Veículos (período selecionado)"):
    import plotly.express as px
    vcol="Numero Veiculo"
    mcol=_detect_motorista_col(df)
    if (vcol not in df.columns) or (mcol is None or mcol not in df.columns):
        st.error("Colunas ausentes para Rotatividade: Numero Veiculo e Motorista"); return

    st.markdown("## " + titulo)
    base=df.copy()

    # Filtro por período, se houver datas
    scol = 'Data Hora Inicio Operacao'
    ecol = 'Data Hora Final Operacao'
    has_dates = (scol in base.columns) and (ecol in base.columns)
    if has_dates:
        base[scol] = pd.to_datetime(base[scol], errors='coerce')
        base[ecol] = pd.to_datetime(base[ecol], errors='coerce')
        _s = base[scol].dropna(); _e = base[ecol].dropna()
        from datetime import date as _date
        start_default = (_s.min().date() if not _s.empty else (_e.min().date() if not _e.empty else _date.today()))
        end_default   = (_e.max().date() if not _e.empty else (_s.max().date() if not _s.empty else _date.today()))
        c1, c2 = st.columns(2)
        with c1:
            dt_ini = st.date_input('Início do período (rotatividade)', value=start_default, format='DD/MM/YYYY', key='rot_dtini')
        with c2:
            dt_fim = st.date_input('Fim do período (rotatividade)', value=end_default, format='DD/MM/YYYY', key='rot_dtfim')
        _ini = pd.Timestamp(dt_ini).normalize()
        _fim = pd.Timestamp(dt_fim).normalize() + pd.Timedelta(days=1)
        # filtra por sobreposição de intervalo
        base = base[ base[scol].notna() & base[ecol].notna() & (base[ecol] > _ini) & (base[scol] < _fim) ]
    else:
        st.caption('Aviso: dataset sem colunas de data/tempo para filtrar período na Rotatividade.')

    # Filtros de veículo e motorista
    base[vcol] = base[vcol].astype(str)
    base[mcol] = base[mcol].astype(str)
    pick_veic = st.multiselect('Veículos', sorted(base[vcol].dropna().unique().tolist()), key='rot_filt_vei')
    pick_mot  = st.multiselect('Motoristas', sorted(base[mcol].dropna().unique().tolist()), key='rot_filt_mot')
    if pick_veic:
        base = base[ base[vcol].isin(pick_veic) ]
    if pick_mot:
        base = base[ base[mcol].isin(pick_mot) ]

    if base.empty:
        st.info('Sem registros no período/filtros selecionados para Rotatividade.')
        return

    # Agrupar motoristas únicos por veículo
    grp=(base[[vcol, mcol]].dropna().astype({vcol:str, mcol:str})
         .drop_duplicates()
         .groupby(vcol)[mcol].nunique().reset_index(name="Motoristas Únicos"))
    grp = grp.sort_values("Motoristas Únicos", ascending=False).rename(columns={vcol:"Veículo"})
    if grp.empty:
        st.info('Sem dados para exibir na Rotatividade com os filtros atuais.')
        return

    grp["Veículo"]=grp["Veículo"].astype(str)
    fig=px.bar(grp, x="Veículo", y="Motoristas Únicos", text="Motoristas Únicos",
               title="Quantidade de motoristas únicos por veículo (período selecionado)")
    fig.update_xaxes(type="category")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # Dados + CSV
    st.markdown("#### Dados — Rotatividade (motoristas únicos por veículo)")
    if not grp.empty:
        st.dataframe(grp, use_container_width=True, hide_index=True)
        _csv = grp.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar CSV – Rotatividade", data=_csv, file_name="rotatividade_motoristas_por_veiculo.csv", mime="text/csv", key="dl_rot_veic")
    else:
        st.caption("Sem dados para exportar neste momento.")

# ==============================
# Painel: Tabela de horas por Motorista × Dia (com HE e negativas)
# ==============================
def show_tabela_horas_motoristas_periodo(df, titulo="🗓️ Tabela de horas por Motorista × Dia (período)"):
    import numpy as np
    vcol = "Numero Veiculo"
    lcol = "Nome Linha"
    scol = "Data Hora Inicio Operacao"
    ecol = "Data Hora Final Operacao"
    mcol = _detect_motorista_col(df)

    missing = [c for c in [mcol, lcol, scol, ecol] if c is None or c not in df.columns]
    if missing:
        st.error("Colunas ausentes para a tabela de horas: " + ", ".join(map(str, missing)))
        return

    st.markdown("## " + titulo)

    base = df.copy()
    base[scol] = pd.to_datetime(base[scol], errors="coerce")
    base[ecol] = pd.to_datetime(base[ecol], errors="coerce")

    sdates = base[scol].dropna().dt.date
    edates = base[ecol].dropna().dt.date
    from datetime import date as _date
    start_default = (sdates.min() if not sdates.empty else (edates.min() if not edates.empty else _date.today()))
    end_default   = (edates.max() if not edates.empty else (sdates.max() if not sdates.empty else _date.today()))

    col1, col2 = st.columns(2)
    with col1:
        d_ini = st.date_input("Início do período", value=start_default, format="DD/MM/YYYY", key="horas_tbl_dtini")
    with col2:
        d_fim = st.date_input("Fim do período", value=end_default, format="DD/MM/YYYY", key="horas_tbl_dtfim")
    if d_fim < d_ini:
        st.warning("A data final é anterior à inicial.")
        return

    dt_ini = pd.Timestamp(d_ini).normalize()
    dt_fim = pd.Timestamp(d_fim).normalize() + pd.Timedelta(days=1)  # exclusivo

    with st.expander("Filtros (opcionais)"):
        mot_list = sorted(base[mcol].astype(str).dropna().unique().tolist())
        pick_mot = st.multiselect("Filtrar Motoristas", mot_list, default=mot_list, key="horas_tbl_filt_mot")
        linhas = sorted(base[lcol].astype(str).dropna().unique().tolist())
        pick_lin = st.multiselect("Filtrar Linhas", linhas, default=linhas, key="horas_tbl_filt_lin")
        veics = sorted(base[vcol].astype(str).dropna().unique().tolist()) if vcol in base.columns else []
        pick_vei = st.multiselect("Filtrar Veículos", veics, default=veics, key="horas_tbl_filt_vei") if veics else []

    mask = (
        base[scol].notna() & base[ecol].notna() &
        (base[ecol] > dt_ini) & (base[scol] < dt_fim) &
        (base[mcol].astype(str).isin(pick_mot)) &
        (base[lcol].astype(str).isin(pick_lin))
    )
    if veics:
        mask &= base[vcol].astype(str).isin(pick_vei)
    base = base.loc[mask, [mcol, lcol, scol, ecol]].copy()
    if base.empty:
        st.info("Sem registros no período/filtros selecionados.")
        return

    # Rateio por dia (clipping)
    segs = []
    for _, r in base.iterrows():
        s = r[scol]; e = r[ecol]
        if pd.isna(s) or pd.isna(e): 
            continue
        s = max(s, dt_ini)
        e = min(e, dt_fim)
        if s >= e:
            continue
        cur = s
        while cur < e:
            next_boundary = (cur.normalize() + pd.Timedelta(days=1))
            e_part = min(e, next_boundary)
            dur_min = (e_part - cur).total_seconds()/60.0
            segs.append({
                "Motorista": str(r[mcol]),
                "Dia": cur.normalize().date(),
                "Minutos": dur_min
            })
            cur = e_part

    if not segs:
        st.info("Sem segmentos no período recortado.")
        return

    seg = pd.DataFrame(segs)
    piv = seg.pivot_table(index="Motorista", columns="Dia", values="Minutos", aggfunc="sum", fill_value=0, observed=False)

    # Garantir todas as datas do período como colunas (mesmo sem trabalho)
    all_days = pd.date_range(dt_ini, dt_fim - pd.Timedelta(days=1), freq="D").date
    piv = piv.reindex(columns=list(all_days), fill_value=0)

    limite = 7*60 + 20  # 440
    total_min = piv.sum(axis=1)
    dias_trab = (piv > 0).sum(axis=1).astype(int)
    dias_nao_trab = (piv == 0).sum(axis=1).astype(int)

    def _has_streak_ge7(vals):
        max_run = run = 0
        for v in (vals > 0).astype(int):
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return max_run >= 7

    trabalhou_ge7 = piv.apply(lambda r: _has_streak_ge7(r.values), axis=1)

    he_min = (piv - limite).clip(lower=0).sum(axis=1)
    neg_min = ((dias_trab * limite) - total_min).clip(lower=0)

    # Ordenar colunas por data (já está) e montar final
    final_num = piv.copy()
    final_num["Dias Trabalhados"] = dias_trab
    final_num["Dias Não Trabalhados"] = dias_nao_trab
    final_num["Trabalhou ≥7 dias seguidos?"] = np.where(trabalhou_ge7, "Sim", "Não")
    final_num["Total (min)"] = total_min
    final_num["HE Total (min)"] = he_min
    final_num["Horas Negativas (min)"] = neg_min

    # Styler
    day_cols = list(piv.columns)
    def color_cells(v):
        colors = []
        for x in v:
            if x > limite:
                colors.append("background-color: #ff6b6b; color: #000")  # vermelho
            elif x < 6*60:
                colors.append("background-color: #ffe082; color: #000")  # amarelo
            else:
                colors.append("background-color: #81c784; color: #000")  # verde
        return colors

    rename_cols = {c: pd.Timestamp(c).strftime("%d/%m") for c in day_cols}
    day_cols_ren = [rename_cols.get(c, c) for c in day_cols]
    display_df = final_num.rename(columns=rename_cols).reset_index()
    # prefixo 🔴 para quem tem streak ≥7
    flag_map = trabalhou_ge7.to_dict()
    display_df["Motorista"] = display_df["Motorista"].astype(str).map(lambda x: "🔴 " + x if flag_map.get(x, False) else x)

    sty = display_df.style
    sty = sty.apply(color_cells, subset=day_cols_ren, axis=1)
    fmt_cols = day_cols_ren + ["Total (min)", "HE Total (min)", "Horas Negativas (min)"]
    sty = sty.format(_fmt_hhmm, subset=fmt_cols)

    st.markdown("#### Tabela de horas (cores: vermelho >07:20, amarelo <06:00, verde entre 06:00–07:20) — 🔴 motorista com sequência ≥7 dias")
    st.write(sty)

    # CSV
    csv_minutes = piv.rename(columns=rename_cols).applymap(_fmt_hhmm)
    csv_out = csv_minutes.copy()
    csv_out["Dias Trabalhados"] = dias_trab
    csv_out["Dias Não Trabalhados"] = dias_nao_trab
    csv_out["Trabalhou ≥7 dias seguidos?"] = np.where(trabalhou_ge7, "Sim", "Não")
    csv_out["Total"] = total_min.map(_fmt_hhmm)
    csv_out["HE Total"] = he_min.map(_fmt_hhmm)
    csv_out["Horas Negativas"] = neg_min.map(_fmt_hhmm)
    csv = csv_out.to_csv(index=True, encoding="utf-8-sig")
    st.download_button("Baixar CSV (horas por motorista × dia)", data=csv, file_name="horas_motoristas_por_dia.csv", mime="text/csv")

# ==============================
# Execução: descobrir DF e renderizar painéis
# ==============================
_DF_CANDS = ['df_scope','df_filtrado','df_filtered','df_periodo','df_period','df_view','df_final','df_result','df_base_filtrado','df']
def _pick_df():
    import pandas as _pd
    for name in _DF_CANDS:
        if name in globals():
            obj = globals()[name]
            if isinstance(obj, _pd.DataFrame) and not obj.empty:
                return obj
    # Se não houver global, tenta ler CSV enviado (se existir)
    return None

df_candidate = _pick_df()

if df_candidate is None:
    st.warning("Nenhum DataFrame detectado automaticamente. Carregue os dados ou verifique os nomes das variáveis.")
else:
    # Ordem sugerida: timelines, rotatividade, tabela de horas
    show_linha_do_tempo_alocacao_1dia(df_candidate)
    show_linha_do_tempo_motoristas_linhas_1dia(df_candidate)
    show_linha_do_tempo_motoristas_veiculos_1dia(df_candidate)
    show_rotatividade_motoristas_veiculos(df_candidate)
    show_tabela_horas_motoristas_periodo(df_candidate)
