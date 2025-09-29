# --- Helper function to normalize Google Ads CSV ---
def normalize_google_ads_csv(uploaded_file):
    """
    Handle messy Google Ads CSV exports, encodings, and normalize columns.
    """
    try:
        # Try reading with different encodings and skip metadata rows
        df = None
        for encoding in ['utf-8', 'utf-8-sig', 'latin1']:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, skip_blank_lines=True)
                
                # Check if first row contains metadata (like "Search terms report")
                if not any("search" in str(col).lower() for col in df.columns):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, skiprows=2, encoding=encoding, skip_blank_lines=True)
                break
            except Exception:
                continue
        
        if df is None:
            raise ValueError("Could not read file with supported encodings")
        
        # Normalize column names (lowercase + stripped)
        df.columns = df.columns.str.strip().str.lower()
        
        # Map Google Ads column variations to standard names
        column_mapping = {
            'search term': 'Search term',
            'search terms': 'Search term',
            'clicks': 'Clicks',
            'impr.': 'Impressions',
            'impressions': 'Impressions',
            'cost': 'Cost',
            'conversions': 'Conversions',
            'all conv.': 'Conversions',
            'conversions (many-per-click)': 'Conversions',
            'conversions (1-per-click)': 'Conversions'
        }
        
        # Apply column mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['Clicks', 'Impressions', 'Cost', 'Conversions']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error processing CSV file: {str(e)}")

import streamlit as st
import pandas as pd
import re
import numpy as np
from io import BytesIO

class SearchQueryMiner:
    def __init__(self, brand_terms=None, competitor_terms=None):
        # Default brand terms (you'd customize this per client)
        self.brand_terms = brand_terms or ['nike', 'nike.com']
        
        # Default competitor terms
        self.competitor_terms = competitor_terms or [
            'adidas', 'puma', 'reebok', 'under armour', 'new balance', 
            'converse', 'vans', 'asics', 'skechers'
        ]
        
        # Negative intent keywords (low commercial value)
        self.negative_intent_keywords = [
            'free', 'cheap', 'discount', 'coupon', 'jobs', 'career', 
            'hiring', 'employee', 'work at', 'fake', 'knockoff', 
            'replica', 'customer service', 'return policy', 'complaint',
            'review', 'vs', 'compare', 'comparison'
        ]
        
        # High-intent keywords (good for expansion)
        self.high_intent_keywords = [
            'buy', 'purchase', 'shop', 'store', 'outlet', 'sale',
            'price', 'cost', 'near me', 'online', 'best'
        ]
    
    def classify_query(self, query):
        """Classify a search query into categories"""
        query_lower = query.lower()
        
        # Brand detection
        is_brand = any(brand in query_lower for brand in self.brand_terms)
        
        # Competitor detection
        is_competitor = any(comp in query_lower for comp in self.competitor_terms)
        
        # Negative intent detection
        has_negative_intent = any(neg in query_lower for neg in self.negative_intent_keywords)
        
        # High intent detection
        has_high_intent = any(high in query_lower for high in self.high_intent_keywords)
        
        # Classification logic
        if has_negative_intent:
            return 'NEGATIVE_INTENT'
        elif is_competitor:
            return 'COMPETITOR'
        elif is_brand and has_high_intent:
            return 'BRAND_HIGH_INTENT'
        elif is_brand:
            return 'BRAND_GENERAL'
        elif has_high_intent:
            return 'HIGH_INTENT'
        else:
            return 'GENERAL'
    
    def calculate_priority_score(self, row):
        """Calculate priority score for each query"""
        clicks = row['Clicks']
        cost = row['Cost']
        conversions = row['Conversions']
        
        # Calculate metrics
        ctr = clicks / row['Impressions'] if row['Impressions'] > 0 else 0
        cpc = cost / clicks if clicks > 0 else 0
        conversion_rate = conversions / clicks if clicks > 0 else 0
        
        # Scoring logic
        score = 0
        
        # High cost with no conversions = high priority for negatives
        if cost > 50 and conversions == 0:
            score += 100
        
        # High clicks with no conversions = medium priority for negatives
        if clicks > 30 and conversions == 0:
            score += 50
        
        # High conversion rate = good for expansion
        if conversion_rate > 0.05:  # 5% conversion rate
            score -= 50  # Negative score means good performance
        
        # High CTR = potentially good
        if ctr > 0.05:  # 5% CTR
            score -= 25
        
        return score
    
    def get_recommendation(self, row, category):
        """Get action recommendation based on category and performance"""
        priority_score = self.calculate_priority_score(row)
        conversions = row['Conversions']
        cost = row['Cost']
        clicks = row['Clicks']
        
        # Recommendation logic
        if category == 'NEGATIVE_INTENT':
            return 'ADD_NEGATIVE', 'Low commercial intent - add as negative keyword'
        
        elif category == 'COMPETITOR':
            if conversions == 0 and cost > 30:
                return 'ADD_NEGATIVE', 'Competitor term with no conversions - consider negative'
            else:
                return 'MONITOR', 'Competitor term - monitor performance'
        
        elif priority_score > 75:
            return 'ADD_NEGATIVE', f'High cost (${cost:.2f}) with poor performance'
        
        elif priority_score < -30 and category in ['BRAND_HIGH_INTENT', 'HIGH_INTENT']:
            return 'ADD_KEYWORD', 'High-performing query - consider adding as keyword'
        
        elif conversions > 0:
            return 'KEEP', 'Converting query - keep monitoring'
        
        else:
            return 'MONITOR', 'Neutral performance - continue monitoring'
    
    def analyze_queries(self, df):
        """Main analysis function"""
        results = []
        
        for idx, row in df.iterrows():
            query = row['Search term']
            category = self.classify_query(query)
            priority_score = self.calculate_priority_score(row)
            action, reason = self.get_recommendation(row, category)
            
            # Calculate key metrics
            ctr = (row['Clicks'] / row['Impressions'] * 100) if row['Impressions'] > 0 else 0
            cpc = row['Cost'] / row['Clicks'] if row['Clicks'] > 0 else 0
            conv_rate = (row['Conversions'] / row['Clicks'] * 100) if row['Clicks'] > 0 else 0
            
            results.append({
                'Search Term': query,
                'Category': category,
                'Clicks': row['Clicks'],
                'Cost': f"${row['Cost']:.2f}",
                'Conversions': row['Conversions'],
                'CTR (%)': f"{ctr:.2f}%",
                'CPC': f"${cpc:.2f}" if cpc > 0 else "$0.00",
                'Conv Rate (%)': f"{conv_rate:.2f}%" if conv_rate > 0 else "0.00%",
                'Priority Score': priority_score,
                'Recommended Action': action,
                'Reason': reason
            })
        
        return pd.DataFrame(results)

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# Streamlit App
def main():
    st.set_page_config(page_title="Search Query Miner", page_icon="🔍", layout="wide")
    
    st.title("🔍 Google Ads Search Query Mining Tool")
    st.markdown("**Automatically analyze your Google Ads search terms and get actionable recommendations**")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Brand terms input
    brand_terms_input = st.sidebar.text_input(
        "Brand Terms (comma-separated)", 
        value="nike, nike.com",
        help="Enter your brand terms separated by commas"
    )
    brand_terms = [term.strip().lower() for term in brand_terms_input.split(',') if term.strip()]
    
    # Competitor terms input
    competitor_terms_input = st.sidebar.text_area(
        "Competitor Terms (comma-separated)", 
        value="adidas, puma, reebok, under armour, new balance",
        help="Enter competitor brand names separated by commas"
    )
    competitor_terms = [term.strip().lower() for term in competitor_terms_input.split(',') if term.strip()]
    
    # File upload
    st.header("📁 Upload Your Google Ads Search Terms Report")
    st.markdown("Export your search terms report from Google Ads as CSV and upload it here.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file", 
        type="csv",
        help="Upload a CSV file with columns: Search term, Clicks, Impressions, Cost, Conversions"
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['Search term', 'Clicks', 'Impressions', 'Cost', 'Conversions']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Required columns: Search term, Clicks, Impressions, Cost, Conversions")
                return
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())
            
            # Initialize miner with custom terms
            miner = SearchQueryMiner(brand_terms=brand_terms, competitor_terms=competitor_terms)
            
            # Analyze queries
            with st.spinner("🔍 Analyzing search queries..."):
                results_df = miner.analyze_queries(df)
            
            # Display results
            st.subheader("🎯 Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_queries = len(results_df)
                st.metric("Total Queries", total_queries)
            
            with col2:
                negative_queries = len(results_df[results_df['Recommended Action'] == 'ADD_NEGATIVE'])
                st.metric("Negative Keywords", negative_queries)
            
            with col3:
                expansion_queries = len(results_df[results_df['Recommended Action'] == 'ADD_KEYWORD'])
                st.metric("Expansion Opportunities", expansion_queries)
            
            with col4:
                # Calculate potential savings
                negative_df = results_df[results_df['Recommended Action'] == 'ADD_NEGATIVE']
                potential_savings = sum([float(cost.replace('$', '')) for cost in negative_df['Cost']])
                st.metric("Potential Savings", f"${potential_savings:.2f}")
            
            # Action breakdown
            st.subheader("📈 Recommended Actions Breakdown")
            action_counts = results_df['Recommended Action'].value_counts()
            st.bar_chart(action_counts)
            
            # Filters
            st.subheader("🔧 Filter Results")
            col1, col2 = st.columns(2)
            
            with col1:
                action_filter = st.selectbox(
                    "Filter by Action",
                    options=['All'] + list(results_df['Recommended Action'].unique())
                )
            
            with col2:
                category_filter = st.selectbox(
                    "Filter by Category",
                    options=['All'] + list(results_df['Category'].unique())
                )
            
            # Apply filters
            filtered_df = results_df.copy()
            if action_filter != 'All':
                filtered_df = filtered_df[filtered_df['Recommended Action'] == action_filter]
            if category_filter != 'All':
                filtered_df = filtered_df[filtered_df['Category'] == category_filter]
            
            # Display filtered results
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download button
            csv = convert_df_to_csv(results_df)
            st.download_button(
                label="📥 Download Full Analysis as CSV",
                data=csv,
                file_name='search_query_analysis.csv',
                mime='text/csv'
            )
            
            # Negative keywords export
            negative_keywords_df = results_df[results_df['Recommended Action'] == 'ADD_NEGATIVE'][['Search Term']]
            negative_keywords_df.columns = ['Keyword']
            negative_csv = convert_df_to_csv(negative_keywords_df)
            
            st.download_button(
                label="🚫 Download Negative Keywords List",
                data=negative_csv,
                file_name='negative_keywords.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your CSV has the required columns: Search term, Clicks, Impressions, Cost, Conversions")
    
    else:
        # Show sample data format
        st.subheader("📋 Expected CSV Format")
        sample_data = {
            'Search term': ['nike running shoes', 'free nike shoes', 'nike store near me'],
            'Clicks': [45, 67, 56],
            'Impressions': [1200, 2100, 1400],
            'Cost': [67.50, 89.30, 78.90],
            'Conversions': [3, 0, 4]
        }
        st.dataframe(pd.DataFrame(sample_data))

if __name__ == "__main__":
    main()
