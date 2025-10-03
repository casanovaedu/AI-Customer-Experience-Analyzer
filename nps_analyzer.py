# nps_analyzer.py

import pandas as pd
import numpy as np
import sys
import os
import time
import re
from transformers import pipeline
import warnings
import textwrap
import random
from datetime import timedelta

warnings.filterwarnings("ignore")

# --- Default Configuration ---
EXECUTIVE_REPORT_PATH = 'reports/executive_summary_report.txt'
EXCEL_OUTPUT_PATH = 'reports/detailed_analysis_results.xlsx'
TARGETED_EXCEL_PATH = 'reports/targeted_deep_dive_data.xlsx'
COL_LOCATOR = 'Nps Locator'
COL_DESTINATION = 'Nps Destination Country Name'
COL_DATE = 'Nps Travel Departure Date'
COL_RETURN_DATE = 'Nps Travel Return Date'
COL_NPS = 'Nps NPS'
FREE_TEXT_COLS = ['Nps Why Exoticca', 'Nps Other Comment']
PAIN_POINT_LABELS = [
    'Guide: Attitude or Unfriendliness', 'Guide: Lack of Knowledge or Poor Language Skills',
    'Itinerary: Pace Too Rushed', 'Itinerary: Not Enough Free Time',
    'Hotel: Poor Location or Neighborhood', 'Hotel: Lack of Cleanliness or Poor Quality',
    'Transport: Flight Delays or Cancellations', 'Transport: Bus Quality or Comfort',
    'Food: Poor Quality or Lack of Variety', 'Communication: Pre-Trip Information',
    'Communication: On-Trip Updates and Changes', 'Value: Unexpected Costs or Not Worth the Price',
    'Activities: Poor Quality or Not as Described'
]

def analyze_feedback_with_ai(df, pain_point_labels, progress_bar, status_text, funny_messages):
    """
    Uses a zero-shot AI model to classify feedback, now with real-time progress reporting
    and time estimation for the Streamlit UI.
    """
    start_time = time.time()
    try:
        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0)
    except Exception:
        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    feedback_to_analyze = df[df['Combined_Feedback'].str.strip() != ''].copy()
    total_reviews = len(feedback_to_analyze)
    if total_reviews == 0:
        status_text.warning("No feedback text found to analyze.")
        df['Pain_Point'] = 'N/A'
        return df
    feedback_list = feedback_to_analyze['Combined_Feedback'].str.slice(0, 1024).tolist()
    results = []
    CHUNK_SIZE = 100
    time_per_chunk = 0
    try:
        for i in range(0, total_reviews, CHUNK_SIZE):
            chunk = feedback_list[i:i + CHUNK_SIZE]
            chunk_start_time = time.time()
            outputs = classifier(chunk, pain_point_labels, multi_label=False, batch_size=16)
            chunk_results = [output['labels'][0] for output in outputs]
            results.extend(chunk_results)
            chunk_end_time = time.time()
            if i == 0:
                time_per_chunk = chunk_end_time - chunk_start_time
            processed_count = len(results)
            progress_percentage = processed_count / total_reviews
            progress_bar.progress(progress_percentage)
            if time_per_chunk > 0:
                chunks_remaining = (total_reviews - processed_count) / CHUNK_SIZE
                time_remaining = chunks_remaining * time_per_chunk
                time_eta_str = time.strftime("%M:%S", time.gmtime(time_remaining))
                funny_message = random.choice(funny_messages)
                status_text.text(f"Analyzed {processed_count}/{total_reviews}... (ETA: {time_eta_str}) - {funny_message}")
            else:
                status_text.text(f"Analyzed {processed_count}/{total_reviews}...")
    except Exception as e:
        status_text.error(f"An error occurred during AI analysis: {e}")
        results.extend(['Analysis Error'] * (total_reviews - len(results)))
    feedback_to_analyze['Pain_Point'] = results
    df = df.merge(feedback_to_analyze[['Pain_Point']], left_index=True, right_index=True, how='left')
    df['Pain_Point'].fillna('No Text Feedback', inplace=True)
    return df


def calculate_nps(series):
    promoters = (series >= 9).sum()
    detractors = (series <= 6).sum()
    total = series.count()
    if total == 0: return 0
    return round(((promoters - detractors) / total) * 100)

def calculate_nps_metrics(series):
    total = series.count()
    if total == 0:
        return {'nps': 0, 'promoters': 0, 'passives': 0, 'detractors': 0, 'count': 0}
    promoters = (series >= 9).sum()
    detractors = (series <= 6).sum()
    passives = total - promoters - detractors
    return {'nps': round(((promoters - detractors) / total) * 100), 'promoters': round((promoters/total)*100), 'passives': round((passives/total)*100), 'detractors': round((detractors/total)*100), 'count': total}

def extract_snippet(full_text, keywords, locator):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)
    for sentence in sentences:
        if any(keyword.lower() in sentence.lower() for keyword in keywords):
            return f'"{textwrap.shorten(sentence, width=150, placeholder="...")}" (Locator: {locator})'
    return f'"{textwrap.shorten(full_text, width=150, placeholder="...")}" (Locator: {locator})'

def get_actionable_item(destination, issue):
    action_plan = {
        'Itinerary: Pace Too Rushed': ("Product Team: Review daily schedules; pilot replacing one activity with a 2-hour 'free time' block.", "Medium"),
        'Hotel: Poor Location or Neighborhood': (f"Contracting Team ({destination}): Re-prioritize hotel selection for ES market to 'central & lively' areas.", "High"),
        'Guide: Lack of Knowledge or Poor Language Skills': (f"Operations ({destination}): Implement stricter language certification for Spanish-speaking guides.", "Medium"),
        'Itinerary: Not Enough Free Time': ("Product Team: Mandate at least one afternoon of 'free time' in all itineraries over 7 days.", "Low"),
        'Food: Poor Quality or Lack of Variety': (f"Operations ({destination}): Audit restaurant partners; include at least one 'authentic local' dining option.", "Medium"),
        'Hotel: Lack of Cleanliness or Poor Quality': (f"Contracting Team ({destination}): Flag all hotels with >10% complaint rate for immediate quality review.", "High"),
        'Default': (f"CX/Ops ({destination}): Form a task force to investigate the root cause of the '{issue}' complaints.", "Medium")
    }
    return action_plan.get(issue, action_plan['Default'])


def generate_executive_summary(df, market_to_analyze, deep_dive_destinations):
    report_parts = ["=======================================================================",
                    f"  EXECUTIVE SUMMARY: {market_to_analyze} vs. Rest of World (RoW) Customer Experience  ",
                    "======================================================================="]

    report_parts.append("\n\n1. EXECUTIVE OVERVIEW\n" + "-"*20)
    
    if market_to_analyze == 'Global (All Markets)':
        global_nps = calculate_nps(df[COL_NPS])
        report_parts.append(f"Global analysis of {len(df)} records shows an overall NPS of {global_nps}%.")
    else:
        market_nps = calculate_nps(df[df['Market_Group'] == market_to_analyze][COL_NPS])
        row_nps = calculate_nps(df[df['Market_Group'] == 'Rest_of_World'][COL_NPS])
        report_parts.append(f"Analysis of {len(df)} records reveals a significant perception gap. The RoW NPS is {row_nps}%, while the {market_to_analyze} market lags at {market_nps}%.")

    report_parts.append(f"\n\n2. DEEP-DIVE ANALYSIS: {', '.join(deep_dive_destinations)}\n" + "-"*40)
    all_actions = {}
    for destination in deep_dive_destinations:
        report_parts.append(f"\n----- {destination.upper()} -----\n")
        dest_df = df[df[COL_DESTINATION] == destination].copy()
        if dest_df.empty:
            report_parts.append("No data available.")
            continue
        
        if market_to_analyze == 'Global (All Markets)':
            dest_nps = calculate_nps(dest_df[COL_NPS])
            report_parts.append(f"Overall NPS Score for {destination}: {dest_nps}%")
        else:
            es_nps = calculate_nps(dest_df[dest_df['Market_Group'] == market_to_analyze][COL_NPS])
            row_nps = calculate_nps(dest_df[dest_df['Market_Group'] == 'Rest_of_World'][COL_NPS])
            report_parts.append(f"NPS Score: {market_to_analyze} Market = {es_nps}%  |  RoW Market = {row_nps}%")
        
        pain_point_summary = dest_df[dest_df['Pain_Point'] != 'No Text Feedback'].groupby('Market_Group')['Pain_Point'].value_counts(normalize=True).unstack().fillna(0) * 100
        if market_to_analyze != 'Global (All Markets)' and market_to_analyze in pain_point_summary.index and 'Rest_of_World' in pain_point_summary.index:
            relative_pain = pd.DataFrame({f'{market_to_analyze}_Percent': pain_point_summary.loc[market_to_analyze], 'RoW_Percent': pain_point_summary.loc['Rest_of_World']})
            relative_pain['Pain_Ratio'] = relative_pain[f'{market_to_analyze}_Percent'] / (relative_pain['RoW_Percent'] + 0.1)
            top_issues = relative_pain[relative_pain[f'{market_to_analyze}_Percent'] > 5].sort_values('Pain_Ratio', ascending=False).head(3)
            
            report_parts.append(f"\nTop Disproportionate Pain Points for {market_to_analyze} Market:")
            for issue, row in top_issues.iterrows():
                report_parts.append(f"\n  - {issue} ({row['Pain_Ratio']:.1f}x more likely):")
                example_review = dest_df[(dest_df['Market_Group'] == market_to_analyze) & (dest_df['Pain_Point'] == issue) & (dest_df['Combined_Feedback'].str.len() > 20)].head(1)
                if not example_review.empty:
                    snippet = extract_snippet(example_review.iloc[0]['Combined_Feedback'], issue.split(':'), example_review.iloc[0][COL_LOCATOR])
                    report_parts.append(f"    Customer Voice: {snippet}")
                
                action, effort = get_actionable_item(destination, issue)
                all_actions[(destination, issue)] = (row['Pain_Ratio'], action, effort)
        elif market_to_analyze == 'Global (All Markets)':
             report_parts.append("\nTop Pain Points (Global):")
             top_global = pain_point_summary.loc['Global'].nlargest(3)
             for issue, perc in top_global.items():
                 report_parts.append(f"  - {issue}: {perc:.1f}% of comments")
        else:
            report_parts.append("\nNot enough comparative data for a full analysis.")

    report_parts.append("\n\n\n3. PRIORITIZED ACTION PLAN\n" + "-"*27)
    sorted_actions = sorted(all_actions.items(), key=lambda item: item[1][0], reverse=True)
    
    actions_by_country = {}
    for (dest, issue), (ratio, action, effort) in sorted_actions:
        if dest not in actions_by_country:
            actions_by_country[dest] = []
        actions_by_country[dest].append(f"  - {action} (Effort: {effort})")
        
    for dest, actions in actions_by_country.items():
        report_parts.append(f"\n{dest.upper()}:")
        report_parts.extend(actions)

    with open(EXECUTIVE_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_parts))
    return "\n".join(report_parts)

def save_detailed_excel_files(df):
    try:
        if 'Market_Group' in df.columns:
            nps_summary = df.groupby('Market_Group')[COL_NPS].agg(['mean', 'count']).round(2)
            pain_point_summary = df[df['Pain_Point'] != 'No Text Feedback'].groupby('Market_Group')['Pain_Point'].value_counts(normalize=True).unstack().fillna(0) * 100
            
            with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Enriched_Raw_Data', index=False)
                nps_summary.to_excel(writer, sheet_name='Overall_NPS_Summary')
                pain_point_summary.to_excel(writer, sheet_name='Overall_Pain_Points')
            
        with pd.ExcelWriter(TARGETED_EXCEL_PATH, engine='openpyxl') as writer:
            pass
    except Exception as e:
        print(f"   ❌ ERROR: Could not save detailed Excel files. Reason: {e}")

def generate_weekly_impact_report(df):
    """
    Analyzes the weekly change in NPS globally, using the RETURN DATE.
    """
    if 'Pain_Point' not in df.columns or COL_RETURN_DATE not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), "Not enough data", {}, pd.Series(dtype='float64')

    if not pd.api.types.is_datetime64_any_dtype(df[COL_RETURN_DATE]):
        df[COL_RETURN_DATE] = pd.to_datetime(df[COL_RETURN_DATE], errors='coerce')
    df.dropna(subset=[COL_RETURN_DATE], inplace=True)
    df['Market'] = df[COL_LOCATOR].str.slice(0, 2)
    
    df['WeekPeriod'] = df[COL_RETURN_DATE].dt.to_period('W-MON')

    weekly_stats = df.groupby('WeekPeriod').agg(
        Global_NPS=(COL_NPS, calculate_nps),
        Detractor_Pct=(COL_NPS, lambda x: calculate_nps_metrics(x)['detractors']),
        Review_Count=(COL_NPS, 'count')
    ).sort_index()
    weekly_stats['NPS_WoW_Change'] = weekly_stats['Global_NPS'].diff()
    
    if weekly_stats.empty or len(weekly_stats) < 2 or weekly_stats['NPS_WoW_Change'].min() >= 0:
        weekly_stats.index = weekly_stats.index.strftime('W%U (%Y-%m-%d)')
        return weekly_stats, pd.DataFrame(), "No significant drop found", {}, pd.Series(dtype='float64')

    biggest_drop_period = weekly_stats['NPS_WoW_Change'].idxmin()
    biggest_drop_week_label = biggest_drop_period.strftime('Week %U (%Y-%m-%d)')
    previous_period = biggest_drop_period - 1
    
    current_df = df[df['WeekPeriod'] == biggest_drop_period]
    previous_df = df[df['WeekPeriod'] == previous_period]
    
    if current_df.empty or previous_df.empty:
        weekly_stats.index = weekly_stats.index.strftime('W%U (%Y-%m-%d)')
        return weekly_stats, pd.DataFrame(), biggest_drop_week_label, {}, pd.Series(dtype='float64')

    top_pain_points_current_week = current_df[current_df['Pain_Point'] != 'No Text Feedback']['Pain_Point'].value_counts(normalize=True).nlargest(5) * 100

    total_current_reviews = len(current_df)
    
    top_5_dests = df[COL_DESTINATION].value_counts().nlargest(5).index.tolist()
    
    current_dest_stats = current_df[current_df[COL_DESTINATION].isin(top_5_dests)].groupby(COL_DESTINATION)[COL_NPS].agg(current_nps=calculate_nps, current_count='count')
    previous_dest_stats = previous_df[previous_df[COL_DESTINATION].isin(top_5_dests)].groupby(COL_DESTINATION)[COL_NPS].agg(previous_nps=calculate_nps, previous_count='count')

    dest_delta = pd.concat([current_dest_stats, previous_dest_stats], axis=1).fillna(0)
    dest_delta['nps_delta'] = dest_delta['current_nps'] - dest_delta['previous_nps']
    
    if total_current_reviews > 0:
        dest_delta['nps_point_impact'] = (dest_delta['nps_delta'] * dest_delta['current_count']) / total_current_reviews
    else:
        dest_delta['nps_point_impact'] = 0
    
    dest_delta = dest_delta.sort_values('nps_point_impact')
    dest_delta = dest_delta.astype({'current_nps': int, 'current_count': int, 'previous_nps': int, 'previous_count': int})
    
    week_dates = {
        "current_week_num": biggest_drop_period.week,
        "previous_week_num": previous_period.week
    }

    weekly_stats.index = weekly_stats.index.strftime('W%U (%Y-%m-%d)')

    return weekly_stats, dest_delta, biggest_drop_week_label, week_dates, top_pain_points_current_week

