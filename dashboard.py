"""
F1ML2 Predictions Dashboard
============================
Streamlit dashboard for F1 qualifying and race predictions

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import xgboost as xgb
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="F1ML2 Predictions",
    page_icon="🏎️",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "quali_training" / "models"
DATA_DIR = BASE_DIR / "quali_training" / "data"

# Title and header
st.title("🏎️ F1ML2 Predictions Dashboard")
st.markdown("---")

# Sidebar for model selection
st.sidebar.title("⚙️ Settings")
prediction_type = st.sidebar.radio(
    "Select Prediction Type:",
    ["Qualifying Predictions", "Race Predictions"]
)

# Helper function to load model
@st.cache_resource
def load_model(model_path):
    """Load XGBoost model"""
    try:
        model = xgb.XGBRanker()
        model.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Helper function to load data
@st.cache_data
def load_data(data_path):
    """Load dataset"""
    try:
        df = pd.read_csv(data_path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Driver name mapping (3-letter codes to full names)
DRIVER_NAMES = {
    'ver': 'Max Verstappen',
    'verstappen': 'Max Verstappen',
    'max_verstappen': 'Max Verstappen',
    'per': 'Sergio Perez',
    'perez': 'Sergio Perez',
    'lec': 'Charles Leclerc',
    'leclerc': 'Charles Leclerc',
    'sai': 'Carlos Sainz',
    'sainz': 'Carlos Sainz',
    'ham': 'Lewis Hamilton',
    'hamilton': 'Lewis Hamilton',
    'rus': 'George Russell',
    'russell': 'George Russell',
    'nor': 'Lando Norris',
    'norris': 'Lando Norris',
    'pia': 'Oscar Piastri',
    'piastri': 'Oscar Piastri',
    'alo': 'Fernando Alonso',
    'alonso': 'Fernando Alonso',
    'str': 'Lance Stroll',
    'stroll': 'Lance Stroll',
    'gas': 'Pierre Gasly',
    'gasly': 'Pierre Gasly',
    'oco': 'Esteban Ocon',
    'ocon': 'Esteban Ocon',
    'bot': 'Valtteri Bottas',
    'bottas': 'Valtteri Bottas',
    'zho': 'Zhou Guanyu',
    'zhou': 'Zhou Guanyu',
    'tsu': 'Yuki Tsunoda',
    'tsunoda': 'Yuki Tsunoda',
    'ric': 'Daniel Ricciardo',
    'ricciardo': 'Daniel Ricciardo',
    'mag': 'Kevin Magnussen',
    'magnussen': 'Kevin Magnussen',
    'kevin_magnussen': 'Kevin Magnussen',
    'hul': 'Nico Hulkenberg',
    'hulkenberg': 'Nico Hulkenberg',
    'alb': 'Alexander Albon',
    'albon': 'Alexander Albon',
    'sar': 'Logan Sargeant',
    'sargeant': 'Logan Sargeant',
    'vet': 'Sebastian Vettel',
    'vettel': 'Sebastian Vettel',
    'rai': 'Kimi Raikkonen',
    'raikkonen': 'Kimi Raikkonen',
    'de_vries': 'Nyck de Vries',
    'lat': 'Nicholas Latifi',
    'msc': 'Mick Schumacher',
}

def get_driver_name(code):
    """Convert driver code to full name"""
    if pd.isna(code):
        return "Unknown"
    code_lower = str(code).lower().strip()
    return DRIVER_NAMES.get(code_lower, code.upper())


# ============================================================================
# QUALIFYING PREDICTIONS
# ============================================================================
if prediction_type == "Qualifying Predictions":
    st.header("🏁 Qualifying Position Predictions")
    st.markdown("Predicts starting grid positions (pole position to P20)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction Results")
        
        # Check if model exists
        quali_model_path = MODELS_DIR / "xgbranker_quali_model.json"
        if not quali_model_path.exists():
            st.error(f"❌ Qualifying model not found at: {quali_model_path}")
            st.info("Please train the model first using: `python quali_training/models/trainmodel_quali.py`")
        else:
            model = load_model(quali_model_path)
            
            if model:
                st.success("✅ Qualifying model loaded successfully!")
                
                # Load sample data for demonstration
                data_path = DATA_DIR / "HOLY_qualifying_v1.csv"
                df = load_data(data_path)
                
                if df is not None:
                    st.info(f"📊 Dataset loaded: {len(df):,} records from {df['season'].min()} to {df['season'].max()}")
                    
                    # Let user select season and round
                    col_select1, col_select2 = st.columns(2)
                    
                    with col_select1:
                        available_seasons = sorted(df['season'].unique(), reverse=True)
                        selected_season = st.selectbox(
                            "Select Season",
                            available_seasons,
                            index=0  # Default to latest season
                        )
                    
                    with col_select2:
                        available_rounds = sorted(df[df['season'] == selected_season]['round'].unique())
                        selected_round = st.selectbox(
                            "Select Round",
                            available_rounds,
                            index=len(available_rounds) - 1  # Default to latest round
                        )
                    
                    st.markdown(f"**Showing:** Season {selected_season}, Round {selected_round}")
                    
                    sample_race = df[(df['season'] == selected_season) & 
                                    (df['round'] == selected_round)].copy()
                    
                    if len(sample_race) > 0:
                        # Prepare features for prediction - drop target, ID columns, and qualifying_secs
                        # Note: Model was trained WITHOUT qualifying_secs (experimental design)
                        cols_to_drop = ['grid', 'driver', 'season', 'round', 'id', 'qualifying_secs']
                        X = sample_race.drop(columns=cols_to_drop, errors='ignore')
                        
                        # Convert boolean columns to int (one-hot encoded features)
                        bool_cols = X.select_dtypes(include=['bool']).columns
                        if len(bool_cols) > 0:
                            X[bool_cols] = X[bool_cols].astype(int)
                        
                        # Drop any remaining non-numeric columns
                        X = X.select_dtypes(include=['number'])
                        
                        # Make prediction
                        try:
                            predictions = model.predict(X)
                            sample_race['pred_score'] = predictions
                            sample_race = sample_race.sort_values('pred_score')
                            
                            # Display predictions with full driver names
                            results_df = sample_race[['driver', 'grid', 'pred_score']].reset_index(drop=True)
                            results_df['driver'] = results_df['driver'].apply(get_driver_name)
                            results_df.index = results_df.index + 1
                            results_df.columns = ['Driver', 'Actual Grid', 'Prediction Score']
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Highlight pole position
                            predicted_pole = get_driver_name(sample_race.iloc[0]['driver'])
                            actual_pole_code = sample_race[sample_race['grid'] == 1]['driver'].values
                            actual_pole = get_driver_name(actual_pole_code[0]) if len(actual_pole_code) > 0 else "N/A"
                            
                            st.markdown("### 🏆 Pole Position")
                            col_a, col_b = st.columns(2)
                            col_a.metric("Predicted Pole", predicted_pole)
                            col_b.metric("Actual Pole", actual_pole)
                            
                            if predicted_pole == actual_pole:
                                st.success("✅ Correct pole prediction!")
                            else:
                                st.warning("❌ Incorrect pole prediction")
                                
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
    
    with col2:
        st.subheader("📊 Model Info")
        st.markdown("""
        **Model Type:** XGBoost Ranker
        
        **Target:** Qualifying grid position
        
        **Prediction Scores:**
        - Lower (more negative) = Better position
        - Scores rank drivers relative to each other
        - Only the order matters, not the actual values
        
        **Features:**
        - Driver/Constructor standings
        - Historical qualifying performance
        - Circuit-specific stats
        - Career poles
        - Weather conditions
        
        **Training Data:**
        - 15,333 qualifying sessions
        - Years: 1983-2021
        """)


# ============================================================================
# RACE PREDICTIONS
# ============================================================================
elif prediction_type == "Race Predictions":
    st.header("🏆 Race Result Predictions")
    st.markdown("Predicts podium finishers (1st, 2nd, 3rd)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Prediction Results")
        
        # Check if model exists
        race_model_path = MODELS_DIR / "xgbranker_model.json"
        if not race_model_path.exists():
            st.error(f"❌ Race model not found at: {race_model_path}")
            st.info("Please train the model first using: `python quali_training/models/trainmodel.py`")
        else:
            model = load_model(race_model_path)
            
            if model:
                st.success("✅ Race model loaded successfully!")
                
                # Load sample data
                data_path = DATA_DIR / "HOLYv1.csv"
                df = load_data(data_path)
                
                if df is not None:
                    st.info(f"📊 Dataset loaded: {len(df):,} records from {df['season'].min()} to {df['season'].max()}")
                    
                    # Let user select season and round
                    col_select1, col_select2 = st.columns(2)
                    
                    with col_select1:
                        available_seasons = sorted(df['season'].unique(), reverse=True)
                        selected_season = st.selectbox(
                            "Select Season",
                            available_seasons,
                            index=0,
                            key="race_season"  # Unique key for race predictions
                        )
                    
                    with col_select2:
                        available_rounds = sorted(df[df['season'] == selected_season]['round'].unique())
                        selected_round = st.selectbox(
                            "Select Round",
                            available_rounds,
                            index=len(available_rounds) - 1,
                            key="race_round"  # Unique key for race predictions
                        )
                    
                    st.markdown(f"**Showing:** Season {selected_season}, Round {selected_round}")
                    
                    sample_race = df[(df['season'] == selected_season) & 
                                    (df['round'] == selected_round)].copy()
                    
                    if len(sample_race) > 0:
                        # Prepare features - drop only target and ID columns
                        cols_to_drop = ['podium', 'driver', 'season', 'round', 'id']
                        X = sample_race.drop(columns=cols_to_drop, errors='ignore')
                        
                        # Convert boolean columns to int
                        bool_cols = X.select_dtypes(include=['bool']).columns
                        if len(bool_cols) > 0:
                            X[bool_cols] = X[bool_cols].astype(int)
                        
                        # Drop any remaining non-numeric columns
                        X = X.select_dtypes(include=['number'])
                        
                        try:
                            predictions = model.predict(X)
                            sample_race['pred_score'] = predictions
                            sample_race = sample_race.sort_values('pred_score')
                            
                            # Display top 10 with full driver names
                            results_df = sample_race[['driver', 'podium', 'pred_score']].head(10).reset_index(drop=True)
                            results_df['driver'] = results_df['driver'].apply(get_driver_name)
                            results_df.index = results_df.index + 1
                            results_df.columns = ['Driver', 'Actual Finish', 'Prediction Score']
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Highlight podium
                            st.markdown("### 🏆 Podium Predictions")
                            pred_podium = [get_driver_name(d) for d in sample_race.head(3)['driver'].tolist()]
                            
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("🥇 1st Place", pred_podium[0] if len(pred_podium) > 0 else "N/A")
                            col_b.metric("🥈 2nd Place", pred_podium[1] if len(pred_podium) > 1 else "N/A")
                            col_c.metric("🥉 3rd Place", pred_podium[2] if len(pred_podium) > 2 else "N/A")
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
    
    with col2:
        st.subheader("📊 Model Info")
        st.markdown("""
        **Model Type:** XGBoost Ranker
        
        **Target:** Race finishing position
        
        **Prediction Scores:**
        - Lower (more negative) = Better predicted finish
        - Scores rank drivers relative to each other
        - Only the order matters, not the actual values
        
        **Features:**
        - Starting grid position
        - Driver/Constructor standings
        - Historical race performance
        - Circuit-specific stats
        - Weather conditions
        - Qualifying time
        
        **Training Data:**
        - 15,363 race results
        - Years: 1983-2022
        """)


else:  # Race Predictions
    pass  # This section is already handled above


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏎️ F1ML2 Predictions Dashboard | Cornell Data Science</p>
    <p>For help, see <code>PIPELINE_GUIDE.md</code></p>
</div>
""", unsafe_allow_html=True)

