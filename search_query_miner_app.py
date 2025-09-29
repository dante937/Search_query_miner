import streamlit as st
import pandas as pd
import re
import numpy as np
from io import BytesIO

class SearchQueryMiner:
    def __init__(self, brand_terms=None, competitor_terms=None):
        self.brand_terms = brand_terms or ['nike', 'nike.com']
        self.competitor_terms = competitor_terms or [
            'adidas', 'puma', 'reebok', 'under armour', 'new balance',
            'converse', 'vans', 'asics', 'skechers'
        ]
        self.negative_intent_keywords = [
            'free','cheap','discount','coupon','jobs','career',
            'hiring','employee','work at','fake','knockoff',
            'replica','customer service','return policy','complaint',
            'review','vs','compare','comparison'
        ]
        self.high_intent_keywords = [
            'buy','purchase','shop','store','outlet','sale',
            'price','cost','near me','online','best'
        ]

    def classify_query(self, query):
        query_lower = query.lower()
        is_brand = any(brand in query_lower for brand in self.brand_terms)
        is_competitor = any(comp in query_lower for comp in self.competitor_terms)
        has_negative_intent = any(neg in query_lower for neg in self.negative_intent_keywords)
        has_high_intent = any(high in query_lower for high in self.high_intent_keywords)

        if has_negative_intent: return 'NEGATIVE_INTENT'
        elif is_competitor: return 'COMPETITOR'
        elif is_brand and has_high_intent: return 'BRAND_HIGH_INTENT'
        elif is_brand: return 'BRAND_GENERAL'
        elif has_high_intent: return 'HIGH_INTENT'
        else: return 'GENERAL'

    def calculate_priority_score(self, row):
        clicks, cost, conversions, impressions = row['clicks'], row['cost'], row['conversions'], row['impressions']
        ctr = clicks / impressions if impressions > 0 else 0
        conversion_rate = conversions / clicks if clicks > 0 else 0
        score = 0
        if cost > 50 and conversions == 0: score += 100
        if clicks > 30 and conversions == 0: score += 50
        if conversion_rate > 0.05: score -= 50
        if ctr > 0.05: score -= 25
        return score

    def get_recommendation(self, row, category):
        priority_score = self.calculate_priority_score(row)
        clicks, cost, conversions = row['clicks'], row['cost'], row['conversions']
        if category == 'NEGATIVE_INTENT':
            return 'ADD_NEGATIVE','Low commercial intent - add as negative keyword'
        if category == 'COMPETITOR':
            if conversions == 0 and cost > 30:
                return 'ADD_NEGATIVE','Competitor term with no conversions - consider negative'
            return 'MONITOR','Competitor term - monitor performance'
        if priority_score > 75:
            return 'ADD_NEGATIVE',f'High cost (${cost:.2f}) with poor performance'
        if priority_score < -30 and category in ['BRAND_HIGH_INTENT','HIGH_INTENT']:
            return 'ADD_KEYWORD','High-performing query - consider adding as keyword'
        if conversions > 0:
            return 'KEEP','Converting query - keep monitoring'
        return 'MONITOR','Neutral performance - continue monitoring'

    def analyze_queries(self, df):
        results = []
        for _, row in df.iterrows():
            query = row['search term']
            category = self.classify_query(query)
            priority_score = self.calculate_priority_score(row)
            action, reason = self.get_recommendation(row, category)
            ctr = (row['clicks']/row['impressions']*100) if row['impressions']>0 else 0
            cpc = row['cost']/row['clicks'] if row['clicks']>0 else 0
            conv_rate = (row['conversions']/row['clicks']*100) if row['clicks']>0 else 0
            results.append({
                'Search Term':query,
                'Category':category,
                'Clicks':row['clicks'],
                'Cost':f"${row['cost']:.2f}",
                'Conversions':row['conversions'],
                'CTR (%)':f"{ctr:.2f}%",
                'CPC':f"${cpc:.2f}" if cpc>0 else "$0.00",
                'Conv Rate (%)':f"{conv_rate:.2f}%",
                'Priority Score':priority_score,
                'Recommended Action':action,
                'Reason':reason
            })
        return pd.DataFrame(results)


# 🔧 Updated Robust Parser with Encoding Handling
def normalize_google_ads_csv(uploaded_file):
    """
    Handle messy Google Ads CSV/TSV exports, encodings, and normalize columns.
    """

    # Step 1: Try different encodings
    df = None
    for enc in ["utf-8", "utf-8-sig", "utf-16"]:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            continue
    if df is None:
        raise ValueError("Could not read file with supported encodings: utf-8, utf-8-sig, utf-16")

    # Step 2: Drop empty cols
    df = df.dropna(axis=1, how="all")

    # Step 3: Retry if first row looks like metadata (date ranges)
    if not any("search term" in str(c).lower() for c in df.columns):
        for enc in ["utf-8", "utf-8-sig", "utf-16"]:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=1, encoding=enc, on_bad_lines="skip")
                break
            except:
                continue

    # Step 4: Normalize headers
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        'search term': 'search term',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'cost': 'cost',
        'conversions': 'conversions',
        'all conv.': 'conversions',
        'conversions (many-per-click)': 'conversions',
        'conversions (1-per-click)': 'conversions'
    }
    df = df.rename(columns={k: v for k,v in rename_map.items() if k in df.columns})

    # Step 5: Validate required columns
    required = ['search term','clicks','impressions','cost','conversions']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Step 6: Ensure numeric
    for col in ['clicks','impressions','cost','conversions']:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df[required]


# -----------------------
# Streamlit App
# -----------------------
def main():
    st.set_page_config(page_title="Search Query Miner", page_icon="🔍", layout="wide")
    st.title("🔍 Google Ads Search Query Mining Tool")
    st.markdown("**Automatically analyze your Google Ads search terms and get actionable recommendations**")

    # Sidebar config
    st.sidebar.header("⚙️ Configuration")
    brand_terms_input = st.sidebar.text_input("Brand Terms (comma-separated)", value="nike, nike.com")
    brand_terms = [t.strip().lower() for t in brand_terms_input.split(',') if t.strip()]
    competitor_terms_input = st.sidebar.text_area(
        "Competitor Terms (comma-separated)",
        value="adidas, puma, reebok, under armour, new balance"
    )
    competitor_terms = [t.strip().lower() for t in competitor_terms_input.split(',') if t.strip()]

    # File upload
    uploaded_file = st.file_uploader("📁 Upload Your Search Terms Report CSV", type=['csv'])
    if uploaded_file:
        try:
            df = normalize_google_ads_csv(uploaded_file)
            st.success("✅ File uploaded and recognized!")
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())

            miner = SearchQueryMiner(brand_terms=brand_terms, competitor_terms=competitor_terms)
            results_df = miner.analyze_queries(df)
            st.subheader("🎯 Analysis Results")
            st.dataframe(results_df)

            # Download
            st.download_button("📥 Download Full Analysis CSV",
                               results_df.to_csv(index=False).encode("utf-8"),
                               "search_query_analysis.csv","text/csv")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Make sure your CSV is exported with columns Search term, Clicks, Impressions, Cost, Conversions.")

if __name__ == "__main__":
    main()
