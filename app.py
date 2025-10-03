# app.py

import streamlit as st
import pandas as pd
import os
import time
import random
import numpy as np
import datetime
import altair as alt
from nps_analyzer import (
    analyze_feedback_with_ai, 
    generate_executive_summary,
    save_detailed_excel_files,
    generate_weekly_impact_report,
    PAIN_POINT_LABELS,
    COL_DESTINATION,
    COL_DATE,
    COL_RETURN_DATE,
    COL_NPS,
    COL_LOCATOR,
    calculate_nps_metrics,
    FREE_TEXT_COLS
)

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="CX Intelligence Platform")

FUNNY_MESSAGES = [
    "Recalibrating the flux capacitor...", "Herding alpacas for data alignment...",
    "Translating ancient travel scrolls...", "Calculating the optimal number of gelato stops...",
    "Ensuring no customers were left behind (mostly)...", "Analyzing the likelihood of being eaten by a grue...",
    "Counting all the passport stamps...", "Brewing coffee for the AI... it gets grumpy otherwise."
]
LOADING_GIF_URL = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExd2Rna2o2ajY1dTB2c3A0Z3Zob3p1cjRpbjZ4Y2Z5dG54a3BmM2JmOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/sS9syJzflrRBUh22Vz/giphy.gif"

# --- Main App UI ---
st.title("💡 CX Intelligence Platform")
st.markdown("Transforming raw customer feedback into a strategic, data-driven executive summary.")

# --- Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Sidebar for User Inputs ---
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=100)
    st.header("⚙️ Analysis Configuration")
    
    st.markdown("### 1. Choose Workflow")
    workflow = st.radio(
        "Select analysis mode:",
        ('Run Full Analysis (from raw CSV)', 'Generate Report Only (from analyzed data)'),
        key="workflow_radio"
    )

    st.markdown("### 2. Upload Your Data")
    if workflow == 'Run Full Analysis (from raw CSV)':
        uploaded_file = st.file_uploader("Upload Raw Customer Feedback CSV", type=["csv"], key="raw_uploader")
    else:
        uploaded_file = st.file_uploader("Upload PREVIOUSLY ANALYZED Data CSV", type=["csv"], key="analyzed_uploader")

# --- Main Content Area ---
if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
else:
    # Load and cache the dataframe in session state
    if st.session_state.df is None or id(uploaded_file) != st.session_state.get('file_id'):
        with st.spinner("Reading and preparing your file..."):
            df_temp = pd.read_csv(uploaded_file)
            if COL_RETURN_DATE in df_temp.columns:
                df_temp[COL_RETURN_DATE] = pd.to_datetime(df_temp[COL_RETURN_DATE], errors='coerce')
            if COL_DATE in df_temp.columns:
                df_temp[COL_DATE] = pd.to_datetime(df_temp[COL_DATE], errors='coerce')
            df_temp.dropna(subset=[COL_RETURN_DATE, COL_DATE], how='any', inplace=True)
            st.session_state.df = df_temp
            st.session_state.file_id = id(uploaded_file)
    df = st.session_state.df

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 3. Choose Analysis Type")
        
        main_analysis_type = st.selectbox("", ("Strategic Deep-Dive", "Weekly Performance Diagnostic"))
        
        if main_analysis_type == "Strategic Deep-Dive":
            st.markdown("---")
            st.markdown("#### Deep-Dive Configuration")
            analysis_mode = st.selectbox("Select Analysis Mode", ("Compare Two Markets", "Global Summary"))
            
            all_markets = sorted(df[COL_LOCATOR].str.slice(0, 2).unique().tolist())
            
            if analysis_mode == "Compare Two Markets":
                primary_market = st.selectbox("Select Primary Market", options=all_markets, index=all_markets.index('ES') if 'ES' in all_markets else 0)
            else:
                primary_market = 'Global (All Markets)'
                
            deep_dive_destinations = st.multiselect("Destinations for Deep-Dive", options=sorted(df[COL_DESTINATION].unique().tolist()), default=["China", "Turkey", "Peru"])
            
            if COL_DATE in df.columns:
                min_date = df[COL_DATE].min().date()
                max_date = df[COL_DATE].max().date()
                date_range = st.date_input("Filter by Departure Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            
            pain_point_focus = st.multiselect("Focus AI Analysis (Optional)", options=PAIN_POINT_LABELS, default=[])

            st.markdown("---")
            deep_dive_button = st.button("🚀 Generate Deep-Dive Report")
        
        else: # Weekly Performance Diagnostic
            st.info("This diagnostic analyzes global weekly performance based on RETURN DATE.")
            st.markdown("---")
            weekly_button = st.button("🔬 Run Weekly Diagnostic")

    # --- ACTION: Run Strategic Deep-Dive ---
    if 'deep_dive_button' in locals() and deep_dive_button:
        start_date, end_date = date_range
        df_filtered = df[(df[COL_DATE].dt.date >= start_date) & (df[COL_DATE].dt.date <= end_date)].copy()
        
        st.info(f"Analyzing {len(df_filtered)} records from {start_date} to {end_date}.")
        
        df_enriched = pd.DataFrame()
        if workflow == 'Run Full Analysis (from raw CSV)':
            processing_container = st.empty()
            with processing_container.container():
                gif_col, text_col = st.columns([1, 4])
                with gif_col: st.image(LOADING_GIF_URL)
                with text_col:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Initializing AI model...")
            
            df_filtered['Combined_Feedback'] = df_filtered[FREE_TEXT_COLS].fillna('').astype(str).agg(' '.join, axis=1)
            active_pain_points = pain_point_focus if pain_point_focus else PAIN_POINT_LABELS
            df_enriched = analyze_feedback_with_ai(df_filtered, active_pain_points, progress_bar, status_text, FUNNY_MESSAGES)
            
            processing_container.empty()
            st.success("AI analysis complete!")
        else:
            if 'Pain_Point' not in df_filtered.columns:
                st.error("Missing 'Pain_Point' column. Please use 'Run Full Analysis' workflow.")
                st.stop()
            df_enriched = df_filtered.copy()
        
        if analysis_mode == 'Compare Two Markets':
            df_enriched['Market_Group'] = np.where(df_enriched[COL_LOCATOR].str.slice(0, 2) == primary_market, primary_market, 'Rest_of_World')
        else:
            df_enriched['Market_Group'] = 'Global'

        with st.spinner('Generating executive summary and reports...'):
            executive_report = generate_executive_summary(df_enriched, primary_market, deep_dive_destinations)
            save_detailed_excel_files(df_enriched)

        st.header("📊 Executive Dashboard")
        exec_dashboard(df_enriched, analysis_mode, primary_market)

    # --- ACTION: Run Weekly Performance Diagnostic ---
    if 'weekly_button' in locals() and weekly_button:
        df_for_change_analysis = st.session_state.df.copy()

        if 'Pain_Point' not in df_for_change_analysis.columns:
            if workflow == 'Generate Report Only (from analyzed data)':
                st.error("This diagnostic requires a pre-analyzed file.")
                st.stop()
            else: 
                st.info("Raw data detected. Running initial AI analysis for the diagnostic report...")
                processing_container = st.empty()
                with processing_container.container():
                    gif_col, text_col = st.columns([1, 4])
                    with gif_col: st.image(LOADING_GIF_URL)
                    with text_col:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                
                df_for_change_analysis['Combined_Feedback'] = df_for_change_analysis[FREE_TEXT_COLS].fillna('').astype(str).agg(' '.join, axis=1)
                df_for_change_analysis = analyze_feedback_with_ai(df_for_change_analysis, PAIN_POINT_LABELS, progress_bar, status_text, FUNNY_MESSAGES)
                
                st.session_state.df = df_for_change_analysis
                processing_container.empty()
                st.success("Initial AI analysis complete!")

        st.header("📉 Weekly Performance Diagnostic")
        st.info("This analysis identifies the biggest drivers of Global NPS change week-over-week, based on return date.")
        
        with st.spinner("Analyzing weekly NPS trends..."):
            weekly_trends, impact_analysis, biggest_drop_week_label, week_dates, top_pain_points = generate_weekly_impact_report(df_for_change_analysis)
        
        st.markdown(f"#### Global Weekly Performance (Week {week_dates.get('current_week_num', 'N/A')} vs. Week {week_dates.get('previous_week_num', 'N/A')})")
        
        col1, col2 = st.columns(2)
        if not weekly_trends.empty and len(weekly_trends) > 1:
            col1.metric(f"Global NPS (Last Week)", f"{weekly_trends.iloc[-1]['Global_NPS']}%", f"{weekly_trends.iloc[-1]['NPS_WoW_Change']:.0f}%")
            col2.metric(f"Global Detractors (Last Week)", f"{weekly_trends.iloc[-1]['Detractor_Pct']}%", f"{weekly_trends.iloc[-1]['Detractor_Pct'] - weekly_trends.iloc[-2]['Detractor_Pct']:.0f}%", delta_color="inverse")

        st.line_chart(weekly_trends[['Global_NPS', 'Detractor_Pct']])
        
        st.markdown("---")
        
        col_impact, col_pain = st.columns(2)

        with col_impact:
            st.subheader(f"Top 5 Destination Impacts for Biggest Drop Week ({biggest_drop_week_label})")
            if impact_analysis.empty:
                st.warning("Not enough data to calculate destination-level impact for the biggest drop week.")
            else:
                st.dataframe(impact_analysis.style.format({
                    'current_nps': '{:.0f}', 'previous_nps': '{:.0f}',
                    'nps_delta': '{:+.0f}', 'nps_point_impact': '{:+.2f} pts'
                }).bar(subset=['nps_point_impact'], align='zero', color=['#d65f5f', '#5fba7d']))

        with col_pain:
            st.subheader(f"Top Pain Points During Drop Week")
            if top_pain_points.empty:
                st.warning("No pain point data for this week.")
            else:
                st.bar_chart(top_pain_points)

def exec_dashboard(df_enriched, analysis_mode, primary_market):
    """Renders the full executive dashboard UI."""
    col1, col2, col3, col4 = st.columns(4)
    if analysis_mode == 'Compare Two Markets':
        market_metrics = calculate_nps_metrics(df_enriched[df_enriched['Market_Group'] == primary_market][COL_NPS])
        row_metrics = calculate_nps_metrics(df_enriched[df_enriched['Market_Group'] == 'Rest_of_World'][COL_NPS])
        col1.metric(f"{primary_market} NPS", f"{market_metrics['nps']}%", f"{market_metrics['nps'] - row_metrics['nps']}% vs. RoW", delta_color="inverse")
        col2.metric(f"RoW NPS", f"{row_metrics['nps']}%", f"{row_metrics['count']} reviews")
        col3.metric(f"{primary_market} Detractors", f"{market_metrics['detractors']}%", f"{market_metrics['detractors'] - row_metrics['detractors']}% vs. RoW", delta_color="inverse")
        col4.metric(f"{primary_market} Promoters", f"{market_metrics['promoters']}%", f"{market_metrics['promoters'] - row_metrics['promoters']}% vs. RoW")
    else: # Global Summary
        global_metrics = calculate_nps_metrics(df_enriched[COL_NPS])
        col1.metric("Global NPS", f"{global_metrics['nps']}%", f"{global_metrics['count']} total reviews")
        col2.metric("Promoters", f"{global_metrics['promoters']}%")
        col3.metric("Passives", f"{global_metrics['passives']}%")
        col4.metric("Detractors", f"{global_metrics['detractors']}%")

    st.markdown("---")
    
    pain_point_summary = df_enriched[df_enriched['Pain_Point'] != 'No Text Feedback'].groupby('Market_Group')['Pain_Point'].value_counts(normalize=True).unstack().fillna(0) * 100
    chart_col, gap_col = st.columns([2, 1])

    if analysis_mode == 'Compare Two Markets' and primary_market in pain_point_summary.index and 'Rest_of_World' in pain_point_summary.index:
        pps_transposed = pain_point_summary.T
        pps_transposed['Gap'] = pps_transposed[primary_market] - pps_transposed['Rest_of_World']
        
        with chart_col:
            st.subheader("Pain Point Comparison")
            chart_data = pps_transposed.drop(columns='Gap').reset_index().melt('Pain_Point', var_name='Market', value_name='Percentage')
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Percentage:Q', title='% of Comments', axis=alt.Axis(format='.0f')),
                y=alt.Y('Pain_Point:N', sort=alt.EncodingSortField(field="Gap", op="max", order='descending'), title=None),
                color=alt.Color('Market:N', scale=alt.Scale(scheme='tableau10'), legend=alt.Legend(title="Market")),
                yOffset='Market:N',
                tooltip=['Pain_Point', 'Market', alt.Tooltip('Percentage:Q', format='.1f')]
            ).properties(title=f"'{primary_market}' market complains more about issues at the top")
            st.altair_chart(chart, use_container_width=True)

        with gap_col:
            st.subheader("Top 3 Gaps")
            top_gaps = pps_transposed.sort_values('Gap', ascending=False).head(3)
            for issue, row in top_gaps.iterrows():
                st.markdown(f"**{issue}**")
                st.markdown(f"Gap: **{row['Gap']:.1f} pts** ({row.get(primary_market, 0):.1f}% in {primary_market} vs. {row.get('Rest_of_World', 0):.1f}% in RoW)")
                st.markdown("---")
    
    elif analysis_mode == 'Global Summary':
         with chart_col:
            st.subheader("Global Top Pain Points")
            if 'Global' in pain_point_summary.index:
                global_pain_points = pain_point_summary.loc['Global'].sort_values(ascending=False)
                st.bar_chart(global_pain_points)
            else:
                st.warning("No 'Global' data to display.")
         with gap_col:
            st.subheader("Top 3 Pain Points")
            if 'Global' in pain_point_summary.index:
                top_global_issues = pain_point_summary.loc['Global'].nlargest(3)
                for issue, percentage in top_global_issues.items():
                    st.markdown(f"**{issue}**")
                    st.markdown(f"**{percentage:.1f}%** of all feedback comments.")
                    st.markdown("---")
            else:
                st.warning("No 'Global' data to display.")
    else:
        st.warning(f"Could not perform a full comparison.")

    # Executive Summary Expander
    # executive_report = generate_executive_summary(...) # This would be generated above
    # with st.expander("📄 View Full Executive Summary Report", expanded=False):
    #     st.text(executive_report)

