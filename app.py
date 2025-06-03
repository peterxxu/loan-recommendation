import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import shap

# --- Constants and Global Variables ---
REGIONAL_HIERARCHY = {
    "usa - west coast": "north america",
    "usa - east coast": "north america",
    "usa - midwest": "north america",
    "usa - south": "north america",
    "western europe": "emea (europe, middle east, africa)",
    "eastern europe": "emea (europe, middle east, africa)",
    "nordics": "emea (europe, middle east, africa)",
    "southeast asia": "apac (asia-pacific)",
    "east asia": "apac (asia-pacific)",
    "south asia": "apac (asia-pacific)",
    "australia/new zealand": "apac (asia-pacific)",
    "middle east & north africa (mena)": "emea (europe, middle east, africa)",
    "sub-saharan africa": "emea (europe, middle east, africa)",
    "canada": "north america",
    "uk & ireland": "emea (europe, middle east, africa)",
    "north america": "global",
    "emea (europe, middle east, africa)": "global",
    "apac (asia-pacific)": "global",
    "latam (latin america)": "global",
    "global": "global"
}

RFP_NUMERICAL_FEATURES = ['company_revenue_last_fy_usd', 'deal_size_usd', 'number_of_applications_received']
RFP_CATEGORICAL_FEATURES = ['company_stage', 'industry_sector', 'region', 'loan_type_requested', 'purpose_of_funds']
LENDER_NUMERICAL_FEATURES = ['preferred_deal_size_min_usd', 'preferred_deal_size_max_usd',
                             'avg_funding_timeline_days', 'indicative_interest_rate_min_pct',
                             'indicative_interest_rate_max_pct', 'historical_success_rate_pct']
LENDER_CATEGORICAL_FEATURES = ['lender_type', 'risk_appetite']
ALL_MODEL_FEATURES = RFP_NUMERICAL_FEATURES + RFP_CATEGORICAL_FEATURES + LENDER_NUMERICAL_FEATURES + LENDER_CATEGORICAL_FEATURES

# --- Helper Functions ---
def parse_and_normalize_list_semicolon(text):
    if pd.isna(text):
        return []
    return [item.strip().lower() for item in str(text).split(';')]

def get_all_sub_regions_iterative(region, hierarchy):
    all_regions = {region}
    queue = [region]
    children_map = {}
    for child, parent in hierarchy.items():
        children_map.setdefault(parent, []).append(child)
    while queue:
        current_region = queue.pop(0)
        direct_children = children_map.get(current_region, [])
        for child in direct_children:
            if child not in all_regions:
                all_regions.add(child)
                queue.append(child)
    return all_regions

# --- Cached Data Loading and Processing Functions ---
@st.cache_data
def load_data():
    try:
        historical_rfps_df = pd.read_csv('data/historical_rfps.csv')
        lender_preferences_df = pd.read_csv('data/lender_preferences.csv')
    except FileNotFoundError:
        st.error("Error: 'data/historical_rfps.csv' or 'data/lender_preferences.csv' not found. Make sure the data directory exists and files are present.")
        return pd.DataFrame(), pd.DataFrame()
    return historical_rfps_df, lender_preferences_df

@st.cache_data
def get_processed_lender_preferences(_lender_df_original, regional_hierarchy):
    if _lender_df_original.empty:
        return pd.DataFrame()
    lender_df = _lender_df_original.copy()
    lender_df['preferred_industries_list'] = lender_df['preferred_industries'].apply(parse_and_normalize_list_semicolon)
    lender_df['preferred_regions_raw'] = lender_df['preferred_regions'].apply(parse_and_normalize_list_semicolon)
    lender_df['preferred_loan_types_list'] = lender_df['preferred_loan_types'].apply(parse_and_normalize_list_semicolon)
    lender_df['preferred_regions_expanded'] = lender_df['preferred_regions_raw'].apply(
        lambda regions: set().union(*[get_all_sub_regions_iterative(r.lower().strip(), regional_hierarchy) for r in regions if pd.notna(r)])
    )
    return lender_df

@st.cache_data
def get_processed_rfps(_rfps_df_original):
    if _rfps_df_original.empty:
        return pd.DataFrame()
    rfps_df = _rfps_df_original.copy()
    rfps_df['industry_sector_normalized'] = rfps_df['industry_sector'].astype(str).str.lower().str.strip()
    rfps_df['region_normalized'] = rfps_df['region'].astype(str).str.lower().str.strip()
    rfps_df['loan_type_requested_normalized'] = rfps_df['loan_type_requested'].astype(str).str.lower().str.strip()
    return rfps_df

@st.cache_data
def create_match_dataframe(_rfps_processed_df, _lender_preferences_processed_df):
    if _rfps_processed_df.empty or _lender_preferences_processed_df.empty:
        return pd.DataFrame()
    
    historical_rfps_df = _rfps_processed_df.copy()
    lender_preferences_df_local = _lender_preferences_processed_df.copy()
    matched_rfps_data = []

    for _, lender in lender_preferences_df_local.iterrows():
        lender_id = lender['lender_id']
        min_deal_size = lender['preferred_deal_size_min_usd']
        max_deal_size = lender['preferred_deal_size_max_usd']
        preferred_industries = set(lender['preferred_industries_list'])
        preferred_regions = set(lender['preferred_regions_expanded'])
        preferred_loan_types = set(lender['preferred_loan_types_list'])

        deal_size_filter = (historical_rfps_df['deal_size_usd'] >= min_deal_size) & \
                           (historical_rfps_df['deal_size_usd'] <= max_deal_size)
        industry_filter = historical_rfps_df['industry_sector_normalized'].apply(lambda x: x in preferred_industries if pd.notna(x) else False)
        region_filter = historical_rfps_df['region_normalized'].apply(lambda x: x in preferred_regions if pd.notna(x) else False)
        loan_type_filter = historical_rfps_df['loan_type_requested_normalized'].apply(lambda x: x in preferred_loan_types if pd.notna(x) else False)
        
        all_filters = deal_size_filter & industry_filter & region_filter & loan_type_filter
        matched_rfps_for_lender = historical_rfps_df[all_filters]
        
        for _, rfp in matched_rfps_for_lender.iterrows():
            matched_rfps_data.append({
                'lender_id': lender_id,
                'rfp_id': rfp['rfp_id'],
                'lender_name': lender['lender_name'],
                'rfp_title': rfp['rfp_title']
            })
    return pd.DataFrame(matched_rfps_data)

@st.cache_data
def get_full_analysis_df(_matched_rfps_df, _historical_rfps_df_processed, _lender_preferences_df_processed):
    if _matched_rfps_df.empty or _historical_rfps_df_processed.empty or _lender_preferences_df_processed.empty:
        return pd.DataFrame()
        
    df_with_rfp_details = _matched_rfps_df.merge(_historical_rfps_df_processed, on='rfp_id', how='left')
    if 'rfp_title_x' in df_with_rfp_details.columns and 'rfp_title_y' in df_with_rfp_details.columns:
        df_with_rfp_details = df_with_rfp_details.drop(columns=['rfp_title_x'])
        df_with_rfp_details = df_with_rfp_details.rename(columns={'rfp_title_y': 'rfp_title'})

    lender_prefs_to_merge = _lender_preferences_df_processed.drop(
        columns=['preferred_industries_list', 'preferred_regions_raw', 'preferred_loan_types_list', 'preferred_regions_expanded'],
        errors='ignore'
    )
    df_full_analysis = df_with_rfp_details.merge(lender_prefs_to_merge, on='lender_id', how='left')

    if 'lender_name_x' in df_full_analysis.columns and 'lender_name_y' in df_full_analysis.columns:
        df_full_analysis = df_full_analysis.drop(columns=['lender_name_y'])
        df_full_analysis = df_full_analysis.rename(columns={'lender_name_x': 'lender_name'})
    return df_full_analysis

@st.cache_data
def create_df_for_ml(_matched_rfps_df, _historical_rfps_df_processed, _lender_preferences_df_processed):
    if _matched_rfps_df.empty or _historical_rfps_df_processed.empty or _lender_preferences_df_processed.empty:
        return pd.DataFrame()
        
    df_ml = _matched_rfps_df.merge(_historical_rfps_df_processed, on='rfp_id', how='left')
    cols_to_drop_norm = ['industry_sector_normalized', 'region_normalized', 'loan_type_requested_normalized']
    df_ml = df_ml.drop(columns=[col for col in cols_to_drop_norm if col in df_ml.columns], errors='ignore')

    lender_features_for_ml = _lender_preferences_df_processed.drop(columns=[
        'preferred_industries_list', 'preferred_regions_raw', 'preferred_loan_types_list', 'preferred_regions_expanded'
        ], errors='ignore')
    df_ml = df_ml.merge(lender_features_for_ml, on='lender_id', how='left')
    
    df_ml['is_awarded_match'] = ((df_ml['deal_status'].str.lower() == 'funded') & \
                                 (df_ml['lender_id'] == df_ml['awarded_lender_id'])).astype(int)
    return df_ml

@st.cache_resource
def train_model(_df_ml_final):
    if _df_ml_final.empty or 'is_awarded_match' not in _df_ml_final.columns:
        st.warning("ML DataFrame is empty or missing target column. Skipping model training.")
        return None

    X_model = _df_ml_final[ALL_MODEL_FEATURES].copy()
    y_model = _df_ml_final['is_awarded_match']

    for col in RFP_CATEGORICAL_FEATURES + LENDER_CATEGORICAL_FEATURES:
        if X_model[col].isnull().sum() > 0:
            X_model[col] = X_model[col].fillna('missing').astype(str)
        else:
            X_model[col] = X_model[col].astype(str)

    neg_count = y_model.value_counts().get(0,0)
    pos_count = y_model.value_counts().get(1,0)
    scale_pos_weight_train = neg_count / pos_count if pos_count > 0 else 1

    num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, RFP_NUMERICAL_FEATURES + LENDER_NUMERICAL_FEATURES),
            ('cat', cat_pipeline, RFP_CATEGORICAL_FEATURES + LENDER_CATEGORICAL_FEATURES)
        ])

    xgb_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', xgb.XGBClassifier(
                                             scale_pos_weight=scale_pos_weight_train,
                                             random_state=42,
                                             use_label_encoder=False, 
                                             eval_metric='logloss' 
                                         ))])
    try:
        xgb_model_pipeline.fit(X_model, y_model)
    except Exception as e:
        st.error(f"Error during model training: {e}")
        # Minimal error logging for brevity in this context
        return None
    
    return xgb_model_pipeline

# --- Plotting Functions ---
@st.cache_data
def get_funded_rfps_data(_df_full_analysis):
    if _df_full_analysis.empty or 'deal_status' not in _df_full_analysis.columns:
        return pd.DataFrame()
    return _df_full_analysis[_df_full_analysis['deal_status'].str.lower() == 'funded'].copy()

@st.cache_data
def get_successful_lenders_profile(_funded_rfps_df, _lender_preferences_processed_df):
    if _funded_rfps_df.empty or 'awarded_lender_id' not in _funded_rfps_df.columns or _lender_preferences_processed_df.empty:
        return pd.DataFrame()

    successful_lender_counts = _funded_rfps_df['awarded_lender_id'].value_counts()
    successful_lenders_df = pd.DataFrame(successful_lender_counts).reset_index()
    successful_lenders_df.columns = ['lender_id', 'funded_rfp_count']
    
    lender_prefs_for_profile_merge = _lender_preferences_processed_df.drop(columns=[
        'preferred_industries_list', 'preferred_regions_raw', 'preferred_loan_types_list', 'preferred_regions_expanded'
    ], errors='ignore')
    
    successful_lenders_profile = successful_lenders_df.merge(
        lender_prefs_for_profile_merge, on='lender_id', how='left'
    )

    if not successful_lenders_profile.empty:
        if 'preferred_deal_size_min_usd' in successful_lenders_profile.columns and \
           'preferred_deal_size_max_usd' in successful_lenders_profile.columns:
            successful_lenders_profile['preferred_deal_size_midpoint'] = \
                (successful_lenders_profile['preferred_deal_size_min_usd'] + successful_lenders_profile['preferred_deal_size_max_usd']) / 2
        
        if 'indicative_interest_rate_min_pct' in successful_lenders_profile.columns and \
           'indicative_interest_rate_max_pct' in successful_lenders_profile.columns:
            successful_lenders_profile['indicative_interest_rate_midpoint'] = \
                (successful_lenders_profile['indicative_interest_rate_min_pct'] + successful_lenders_profile['indicative_interest_rate_max_pct']) / 2
    
    return successful_lenders_profile

# --- Recommendation Function ---
def recommend_lenders_streamlit(rfp_data_series, _lender_preferences_processed, model_pipeline, top_n=5):
    if _lender_preferences_processed.empty or model_pipeline is None:
        st.warning("Lender data or model not available for recommendations.")
        return pd.DataFrame()

    rfp_deal_size = rfp_data_series.get('deal_size_usd')
    rfp_industry_normalized = str(rfp_data_series.get('industry_sector', 'missing')).lower().strip()
    rfp_region_normalized = str(rfp_data_series.get('region', 'missing')).lower().strip()
    rfp_loan_type_normalized = str(rfp_data_series.get('loan_type_requested', 'missing')).lower().strip()
    
    eligible_lenders_list = []
    for _, lender in _lender_preferences_processed.iterrows():
        min_deal_size = lender['preferred_deal_size_min_usd']
        max_deal_size = lender['preferred_deal_size_max_usd']
        preferred_industries = set(lender['preferred_industries_list'])
        preferred_regions = set(lender['preferred_regions_expanded'])
        preferred_loan_types = set(lender['preferred_loan_types_list'])

        if not (min_deal_size <= rfp_deal_size <= max_deal_size):
            continue
        if rfp_industry_normalized not in preferred_industries:
            continue
        if rfp_region_normalized not in preferred_regions:
            continue
        if rfp_loan_type_normalized not in preferred_loan_types:
            continue
        eligible_lenders_list.append(lender['lender_id'])
    
    if not eligible_lenders_list:
        st.info("No lenders found that meet all hard-filtered preferences for this RFP.")
        return pd.DataFrame()

    eligible_lenders_df = _lender_preferences_processed[_lender_preferences_processed['lender_id'].isin(eligible_lenders_list)].copy()
    if eligible_lenders_df.empty:
        st.info("No eligible lenders after filtering.")
        return pd.DataFrame()
        
    # Prepare input for model
    # Create a DataFrame from rfp_data_series, replicating it for each eligible lender
    rfp_df_repeated = pd.DataFrame([rfp_data_series] * len(eligible_lenders_df))
    
    # Reset index of eligible_lenders_df to align for concatenation
    eligible_lenders_df_reset = eligible_lenders_df.reset_index(drop=True)
    
    # Combine RFP data with lender data
    # Ensure column names match ALL_MODEL_FEATURES for RFP and Lender sections
    combined_features_for_pred = pd.DataFrame()
    for rfp_col in RFP_NUMERICAL_FEATURES + RFP_CATEGORICAL_FEATURES:
        combined_features_for_pred[rfp_col] = rfp_df_repeated[rfp_col].values
        
    for lender_col in LENDER_NUMERICAL_FEATURES + LENDER_CATEGORICAL_FEATURES:
         combined_features_for_pred[lender_col] = eligible_lenders_df_reset[lender_col].values


    # Ensure all columns are present and in correct order, fill missing if any (should not happen if inputs are correct)
    combined_features_for_pred = combined_features_for_pred.reindex(columns=ALL_MODEL_FEATURES)
    
    # Fill NaNs that might have resulted from reindexing or missing data in rfp_data_series
    # (This is a safeguard; ideally, input forms should ensure all necessary data is provided)
    for col in RFP_CATEGORICAL_FEATURES + LENDER_CATEGORICAL_FEATURES:
        if combined_features_for_pred[col].isnull().sum() > 0:
            combined_features_for_pred[col] = combined_features_for_pred[col].fillna('missing').astype(str)
        else:
            combined_features_for_pred[col] = combined_features_for_pred[col].astype(str)


    for col in RFP_NUMERICAL_FEATURES + LENDER_NUMERICAL_FEATURES:
         if combined_features_for_pred[col].isnull().sum() > 0: # Median imputation is in pipeline, but NaNs here could break things
             # This should ideally be handled by how rfp_data_series is constructed (e.g. default values from form)
             # For now, let's fill with a placeholder like 0 or mean, though pipeline imputer should handle it.
             # To be safe for predict_proba, let's ensure no NaNs before preprocessor if not handled robustly by SimpleImputer.
             # SimpleImputer in the pipeline *should* handle this for numerical.
             pass


    predicted_probabilities = model_pipeline.predict_proba(combined_features_for_pred)[:, 1]
    
    eligible_lenders_df['predicted_funding_probability'] = predicted_probabilities
    ranked_lenders = eligible_lenders_df.sort_values(by='predicted_funding_probability', ascending=False)
    
    display_cols = ['lender_name', 'lender_type', 'predicted_funding_probability',
                    'historical_success_rate_pct', 'preferred_deal_size_min_usd',
                    'preferred_deal_size_max_usd', 'risk_appetite']
    
    # Ensure display_cols exist in ranked_lenders
    display_cols_present = [col for col in display_cols if col in ranked_lenders.columns]
    return ranked_lenders[display_cols_present].head(top_n)

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("RFP Insights and Lender Recommendation App")

# --- Load and Prepare Data ---
historical_rfps_df_orig, lender_preferences_df_orig = load_data()

if historical_rfps_df_orig.empty or lender_preferences_df_orig.empty:
    st.stop()

lender_preferences_processed_df = get_processed_lender_preferences(lender_preferences_df_orig, REGIONAL_HIERARCHY)
historical_rfps_processed_df = get_processed_rfps(historical_rfps_df_orig)
matched_rfps_df = create_match_dataframe(historical_rfps_processed_df, lender_preferences_processed_df)
df_full_analysis = get_full_analysis_df(matched_rfps_df, historical_rfps_processed_df, lender_preferences_processed_df)
df_ml_final = create_df_for_ml(matched_rfps_df, historical_rfps_processed_df, lender_preferences_processed_df)
model_pipeline = train_model(df_ml_final)

# --- App Sections ---
tab_titles = ["Data Overview & EDA", "Model Insights", "Lender Recommendation", "Methodology & Improvements"]
if model_pipeline is None: # If model training failed, don't show model-dependent tabs or adjust content
    tab_titles = ["Data Overview & EDA", "Methodology & Improvements"]
    st.warning("Model training failed. Some features of the app will be unavailable.")

tabs = st.tabs(tab_titles)

with tabs[0]: # Data Overview & EDA
    st.header("Exploratory Data Analysis")
    
    with st.expander("Raw Data Overview", expanded=False):
        st.subheader("Historical RFPs (Sample)")
        st.dataframe(historical_rfps_df_orig.head())
        st.subheader("Lender Preferences (Sample)")
        st.dataframe(lender_preferences_df_orig.head())
        st.subheader("Matched RFP-Lender Pairs (Initial Hard Filters - Sample)")
        st.dataframe(matched_rfps_df.head())

    st.subheader("Insights from Funded RFPs")
    funded_rfps_df = get_funded_rfps_data(df_full_analysis)
    if not funded_rfps_df.empty:
        col1_rfp, col2_rfp = st.columns(2)
        with col1_rfp:
            st.image("assets/funded_rfps_top_industries.png", caption="Top 10 Industries for Funded RFPs", use_column_width=True)
            st.image("assets/funded_rfps_deal_sizes.png", caption="Deal Sizes for Funded RFPs", use_column_width=True)
        with col2_rfp:
            st.image("assets/funded_rfps_regions.png", caption="Top 10 Regions for Funded RFPs", use_column_width=True)
            st.image("assets/funded_rfps_loan_types.png", caption="Top 10 Loan Types for Funded RFPs", use_column_width=True)
    else:
        st.write("Data for Funded RFP insights not available or images not found.")

    st.subheader("Profiles of Successful Lenders")
    successful_lenders_profile_df = get_successful_lenders_profile(funded_rfps_df, lender_preferences_processed_df)
    if not successful_lenders_profile_df.empty:
        col1_ldr, col2_ldr = st.columns(2)
        with col1_ldr:
            st.image("assets/lenders_top_types.png", caption="Lender Types (Top 10 Lenders)", use_column_width=True)
            st.image("assets/lenders_funding_timeline.png", caption="Avg. Funding Timeline (Days)", use_column_width=True)
            st.image("assets/lenders_deal_size_midpoints.png", caption="Preferred Deal Size Midpoints", use_column_width=True)
        with col2_ldr:
            st.image("assets/lenders_risk_appetite.png", caption="Risk Appetite of Lenders", use_column_width=True)
            st.image("assets/lenders_interest_rates.png", caption="Indicative Interest Rate Midpoints", use_column_width=True)
    else:
        st.write("Data for Successful Lender profiles not available or images not found.")
        
    st.subheader("RFP-Lender Preference Alignment")
    if not df_full_analysis.empty: 
        col1_pref, col2_pref = st.columns(2)
        with col1_pref:
            st.image("assets/preference_alignment.png", caption="Alignment of Funded RFPs with Awarded Lender Preferences", use_column_width=True)
        # col2_pref can be used for another plot later if needed or left empty to constrain width
    else:
        st.write("Data for Preference Alignment not available or image not found.")
        
if model_pipeline: # Model Insights Tab
    with tabs[1]:
        st.header("Model Insights (Predicting Successful RFP-Lender Matches)")
        st.write("The model predicts if a potential match (RFP + Lender that passed hard filters) would result in successful funding.")

        col1_model, col2_model = st.columns(2)

        with col1_model:
            st.subheader("Feature Importances")
            st.image("assets/ml_feature_importances.png", caption="Top Feature Importances from XGBoost Model", use_column_width=True)
        
        with col2_model:
            st.subheader("SHAP Value Summary Plot")
            st.image("assets/ml_shap_bar.png", caption="SHAP Bar Plot: Feature Impact on Funding Probability", use_column_width=True)
            
            with st.expander("SHAP Dot Summary Plot (Detailed)", expanded=False):
                st.image("assets/ml_shap_dot.png", caption="SHAP Dot Plot: Feature Impact on Funding Probability", use_column_width=True)

if model_pipeline: # Lender Recommendation Tab
    with tabs[2]:
        st.header("Lender Recommendation Engine")
        st.write("Enter RFP details to get lender recommendations. The model will predict funding likelihood for lenders who match basic criteria.")

        # Collect unique values for select boxes safely
        company_stage_options = list(historical_rfps_df_orig['company_stage'].dropna().unique()) if 'company_stage' in historical_rfps_df_orig else []
        industry_options = list(historical_rfps_df_orig['industry_sector'].dropna().unique()) if 'industry_sector' in historical_rfps_df_orig else []
        region_options = list(historical_rfps_df_orig['region'].dropna().unique()) if 'region' in historical_rfps_df_orig else []
        loan_type_options = list(historical_rfps_df_orig['loan_type_requested'].dropna().unique()) if 'loan_type_requested' in historical_rfps_df_orig else []
        purpose_options = list(historical_rfps_df_orig['purpose_of_funds'].dropna().unique()) if 'purpose_of_funds' in historical_rfps_df_orig else []
        
        # Default values for recommendation form - aiming for a known good match
        default_rfp_data = None
        if not df_ml_final.empty and 'is_awarded_match' in df_ml_final.columns:
            awarded_matches = df_ml_final[df_ml_final['is_awarded_match'] == 1]
            if not awarded_matches.empty:
                # Try to get a sample RFP that was awarded and also exists in original RFPs for all details
                # We need rfp_id to merge back and get original, non-normalized values if needed by form
                sample_awarded_rfp_id = awarded_matches.sample(1)['rfp_id'].iloc[0]
                default_rfp_details_series = historical_rfps_df_orig[historical_rfps_df_orig['rfp_id'] == sample_awarded_rfp_id].iloc[0]
                default_rfp_data = {
                    'company_stage': default_rfp_details_series.get('company_stage'),
                    'industry_sector': default_rfp_details_series.get('industry_sector'),
                    'region': default_rfp_details_series.get('region'),
                    'loan_type_requested': default_rfp_details_series.get('loan_type_requested'),
                    'purpose_of_funds': default_rfp_details_series.get('purpose_of_funds'),
                    'company_revenue_last_fy_usd': default_rfp_details_series.get('company_revenue_last_fy_usd', 100000.0),
                    'deal_size_usd': default_rfp_details_series.get('deal_size_usd', 500000.0),
                    'number_of_applications_received': int(default_rfp_details_series.get('number_of_applications_received', 5))
                }

        # Fallback default values if no good match found or data missing
        default_company_stage = default_rfp_data['company_stage'] if default_rfp_data and default_rfp_data['company_stage'] in company_stage_options else (company_stage_options[0] if company_stage_options else None)
        default_industry = default_rfp_data['industry_sector'] if default_rfp_data and default_rfp_data['industry_sector'] in industry_options else (industry_options[0] if industry_options else None)
        default_region = default_rfp_data['region'] if default_rfp_data and default_rfp_data['region'] in region_options else (region_options[0] if region_options else None)
        default_loan_type = default_rfp_data['loan_type_requested'] if default_rfp_data and default_rfp_data['loan_type_requested'] in loan_type_options else (loan_type_options[0] if loan_type_options else None)
        default_purpose = default_rfp_data['purpose_of_funds'] if default_rfp_data and default_rfp_data['purpose_of_funds'] in purpose_options else (purpose_options[0] if purpose_options else None)
        
        default_revenue = default_rfp_data['company_revenue_last_fy_usd'] if default_rfp_data and pd.notna(default_rfp_data['company_revenue_last_fy_usd']) else (historical_rfps_df_orig['company_revenue_last_fy_usd'].median() if 'company_revenue_last_fy_usd' in historical_rfps_df_orig and not historical_rfps_df_orig['company_revenue_last_fy_usd'].dropna().empty else 100000.0)
        default_deal_size = default_rfp_data['deal_size_usd'] if default_rfp_data and pd.notna(default_rfp_data['deal_size_usd']) else (historical_rfps_df_orig['deal_size_usd'].median() if 'deal_size_usd' in historical_rfps_df_orig and not historical_rfps_df_orig['deal_size_usd'].dropna().empty else 500000.0)
        default_apps_received = int(default_rfp_data['number_of_applications_received']) if default_rfp_data and pd.notna(default_rfp_data['number_of_applications_received']) else (int(historical_rfps_df_orig['number_of_applications_received'].median()) if 'number_of_applications_received' in historical_rfps_df_orig and not historical_rfps_df_orig['number_of_applications_received'].dropna().empty else 5)


        with st.form("recommendation_form"):
            st.subheader("RFP Details:")
            col1, col2, col3 = st.columns(3)
            with col1:
                rfp_company_stage = st.selectbox("Company Stage", options=company_stage_options, index=company_stage_options.index(default_company_stage) if default_company_stage and company_stage_options else 0)
                rfp_industry = st.selectbox("Industry Sector", options=industry_options, index=industry_options.index(default_industry) if default_industry and industry_options else 0)
                rfp_region = st.selectbox("Region", options=region_options, index=region_options.index(default_region) if default_region and region_options else 0)
            with col2:
                rfp_loan_type = st.selectbox("Loan Type Requested", options=loan_type_options, index=loan_type_options.index(default_loan_type) if default_loan_type and loan_type_options else 0)
                rfp_purpose = st.selectbox("Purpose of Funds", options=purpose_options, index=purpose_options.index(default_purpose) if default_purpose and purpose_options else 0)
                rfp_company_revenue = st.number_input("Company Revenue (Last FY USD)", min_value=0.0, value=float(default_revenue), step=10000.0, format="%.2f")
            with col3:
                rfp_deal_size = st.number_input("Deal Size (USD)", min_value=0.0, value=float(default_deal_size), step=10000.0, format="%.2f")
                rfp_apps_received = st.number_input("Number of Applications Already Received (Estimate)", min_value=0, value=int(default_apps_received), step=1)
            
            submit_button = st.form_submit_button(label="Get Recommendations")

        if submit_button:
            if not all([rfp_company_stage, rfp_industry, rfp_region, rfp_loan_type, rfp_purpose]):
                 st.error("Please select values for all categorical fields.")
            else:
                rfp_input_data = pd.Series({
                    'company_revenue_last_fy_usd': rfp_company_revenue,
                    'deal_size_usd': rfp_deal_size,
                    'number_of_applications_received': rfp_apps_received,
                    'company_stage': rfp_company_stage,
                    'industry_sector': rfp_industry,
                    'region': rfp_region,
                    'loan_type_requested': rfp_loan_type,
                    'purpose_of_funds': rfp_purpose
                })
                
                with st.spinner("Finding recommendations..."):
                    recommendations = recommend_lenders_streamlit(rfp_input_data, lender_preferences_processed_df, model_pipeline)
                
                if not recommendations.empty:
                    st.subheader("Top Lender Recommendations:")
                    st.dataframe(recommendations.style.format({'predicted_funding_probability': "{:.2%}"}))
                else:
                    st.info("No suitable lenders found based on the hard filters or model predictions for the provided criteria.")

# Methodology Tab (always show if possible)
methodology_tab_index = 3 if model_pipeline else 1
with tabs[methodology_tab_index]:
    st.header("Methodology & Future Directions")
    st.subheader("Top Factors for Successful Funding")
    st.markdown("""
    The analysis below integrates both RFP and lender characteristics to provide a more holistic view of what drives funding success. However, we can narrow the scope to focus solely on RFP attributes based on business needs.
    - At a minimum, hard filters must be met: there should be at least one lender whose preferred deal size, industry, region (with hierarchy), and loan type align with the RFP profile.
    - Beyond the hard filters, the top predictive features include:
        - `Lender risk appetite (Venture)`: RFPs matched with lenders who have a venture-level risk appetite are more likely to be funded, suggesting higher tolerance for risk is a key success factor.
        - `Region (Global)`: RFPs targeting a global region show higher funding probability, consistently ranked as a top factor across both feature importance and SHAP impact.
        - `Loan Type (Strategic Investment)`: Strategic investment requests are positively associated with funding success in both rankings and model explanations.

    Many additional features also contribute meaningfully, and we can tailor our focus depending on the specific use case or decision context.
    """)

    st.subheader("Approach and Model")
    st.markdown("""
    The first step involves hard filtering based on lender preferences as instructed. After applying these filters, I structured the analysis as a machine learning classification task, using both RFP and lender attributes as inputs and modeling the output as whether an RFP was successfully funded (with the assumption that successful fund means both status is funded and there's an awarded lender associated).

    Some approaches I considered include rule-based engine and heuristic ranking algorithms, but I choose machine learning, in particular XGBoost, due to its flexibility and potential in understanding nuances in complicated data. In terms of ML model selection, I choose XGBoost for its performance, robustness, interpretability, and ease of development. 
    """)

    st.subheader("Potential Future Improvements")
    st.markdown("""
    For deeper analysis, the most promising area would be incorporating natural language features like `rfp_description` and `specialization_notes`, and I'll expand on in the next section. Beyond that, I would explore more advanced feature engineering, including time-series patterns and some more nuanced ratio calculations. The machine learning model can also be further improved through techniques like hyperparameter tuning, cross-validation.

    In terms of business needs and production uses, all the code above can be made more modular for reusability and scalability. For example, using object-oriented structure and exposing key components like model inference via APIs to improve reusability and integration with other systems. While not fully demonstrated here, robust error handling and data validation are also critical for deployment. In a production setting, I would implement a structured pipeline to catch and log exceptions and validate data inputs at each stage to ensure stability and maintainability.
    """)

    st.subheader("Using `specialization_notes`")
    st.markdown("""
    Proper handling of this column using natural language processing techniques could reveal some niche preferences and exclusions that allows deeper contextual understanding into lender preferences. Additionally, it can be combined with `rfp_description` via semantic similarity or LLM APIs to further help accurate RFP recommendation. 

    If using traditional NLP approaches, some challenges include proper cleaning and preprocessing to retain meaningful content while filtering out noise.

    If using LLM-based approaches, we would have a more robust handling of unstructured language and better semantic understanding. However, we need to carefully setup validation checks to mitigate halluciation and perform data protections to safeguard sensitive infomration.
    """)

# To run the Streamlit app, save this code as app.py and run `streamlit run app.py` in your terminal.
# Ensure 'data/historical_rfps.csv' and 'data/lender_preferences.csv' are in a 'data' subdirectory.
