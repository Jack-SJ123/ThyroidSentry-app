"""
Thyroid Cancer Recurrence Prediction System
============================================
A comprehensive Streamlit application with 4 pages:
1. Overview - Project introduction and dataset summary
2. EDA - Exploratory Data Analysis with interactive visualizations
3. Model Performance - Comparison of 5 ML models (3 trained + 2 pre-trained)
4. Prediction - Real-time recurrence risk prediction

Features:
- 3 Trained Models: Random Forest, XGBoost, Gradient Boosting
- 2 Pre-trained Models: Best Thyroid Model, Best Tuned Thyroid Model
- Beautiful, modern UI with interactive charts
- Real-time predictions with confidence scores
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Thyroid_Diff.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL1_PATH = MODEL_DIR / "best_thyroid_model.pkl"
MODEL2_PATH = MODEL_DIR / "best_tuned_thyroid_model.pkl"

st.set_page_config(
    page_title="Thyroid Cancer AI | Recurrence Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 2rem;
    }
    .stat-card {
        padding: 1.2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #f9fafb 0%, #eef2ff 100%);
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }
    .stat-caption {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 0.3rem;
    }
    .sidebar-title {
        font-weight: 700;
        font-size: 1rem;
        color: #111827;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.12);
    }
</style>
""", unsafe_allow_html=True)

# ==================== Data Loading ====================

def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def preprocess_data(df):
    """Preprocess data for ML models"""
    df = df.copy()
    
    # Convert Recurred to binary
    if 'Recurred' not in df.columns:
        st.error("Target column 'Recurred' not found in dataset.")
        return pd.DataFrame(), None, {}, []
    
    # Normalize target values: Yes/No to 1/0 where applicable
    if df['Recurred'].dtype == 'O':
        df['Recurred'] = df['Recurred'].replace({
            'Yes': 1, 'yes': 1, 'YES': 1,
            'No': 0, 'no': 0, 'NO': 0
        })
    
    # Separate features and target
    target = 'Recurred'
    
    # Select features (excluding target)
    feature_cols = [col for col in df.columns if col != target]
    
    X = df[feature_cols].copy()
    y_series = df[target].copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Convert y to a writable numpy array
    # Create a new array to ensure it's writable
    y = np.array(y_series.values, dtype=y_series.dtype, copy=True)
    
    return X, y, le_dict, feature_cols

@st.cache_resource
def load_pre_trained_models():
    """Load the two pre-trained models from pickle files"""
    models = {}
    model_info = {}
    
    model_paths = {
        "Best Thyroid Model": MODEL1_PATH,
        "Best Tuned Thyroid Model": MODEL2_PATH
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
                model_info[name] = {"path": path, "loaded": True, "error": None}
            except Exception as e:
                model_info[name] = {"path": path, "loaded": False, "error": str(e)}
        else:
            model_info[name] = {"path": path, "loaded": False, "error": "File not found"}
    
    return models, model_info

# ==================== Utility Functions ====================

def train_models(X_train, y_train, X_test, y_test):
    """Train 3 different ML models and return them"""
    models = {}
    results = {}
    
    # 1. Random Forest
    st.info("🌲 Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    try:
        rf_roc_auc = roc_auc_score(y_test, rf_proba)
    except Exception:
        rf_roc_auc = None
    
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, zero_division=0),
        'recall': recall_score(y_test, rf_pred, zero_division=0),
        'f1': f1_score(y_test, rf_pred, zero_division=0),
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'y_test': y_test,
        'X_test': X_test,
        'roc_auc': rf_roc_auc
    }
    
    # 2. XGBoost
    st.info("🚀 Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    try:
        xgb_roc_auc = roc_auc_score(y_test, xgb_proba)
    except Exception:
        xgb_roc_auc = None
    
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_pred, zero_division=0),
        'predictions': xgb_pred,
        'probabilities': xgb_proba,
        'y_test': y_test,
        'X_test': X_test,
        'roc_auc': xgb_roc_auc
    }
    
    # 3. Gradient Boosting
    st.info("📈 Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    gb_pred = gb_model.predict(X_test)
    gb_proba = gb_model.predict_proba(X_test)[:, 1]
    
    try:
        gb_roc_auc = roc_auc_score(y_test, gb_proba)
    except Exception:
        gb_roc_auc = None
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, zero_division=0),
        'recall': recall_score(y_test, gb_pred, zero_division=0),
        'f1': f1_score(y_test, gb_pred, zero_division=0),
        'predictions': gb_pred,
        'probabilities': gb_proba,
        'y_test': y_test,
        'X_test': X_test,
        'roc_auc': gb_roc_auc
    }
    
    return models, results

def evaluate_model_on_data(model, X_test, y_test, model_name):
    """Evaluate a pre-trained model on test data"""
    try:
        # PyCaret models need DataFrames with column names
        # Ensure X_test is a DataFrame if it's not already
        if not isinstance(X_test, pd.DataFrame):
            # Try to get feature columns from session state
            feature_cols = st.session_state.get('feature_cols', None)
            if feature_cols and len(feature_cols) == X_test.shape[1]:
                X_test = pd.DataFrame(X_test, columns=feature_cols)
            else:
                # Use generic column names
                X_test = pd.DataFrame(X_test, columns=[f'Feature_{i}' for i in range(X_test.shape[1])])
        
        predictions = model.predict(X_test)
        
        # Convert string predictions to numeric if needed
        # PyCaret models may return 'No'/'Yes' instead of 0/1
        if isinstance(predictions[0], str):
            # Map string labels to numeric
            label_map = {
                'No': 0, 'no': 0, 'NO': 0,
                'Yes': 1, 'yes': 1, 'YES': 1
            }
            predictions = np.array([label_map.get(pred, 0) for pred in predictions])
        elif isinstance(predictions[0], (bool, np.bool_)):
            # Convert boolean to int
            predictions = predictions.astype(int)
        
        # Ensure predictions are numpy array of integers
        predictions = np.array(predictions).astype(int)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X_test)
                # Handle different probability formats
                if proba.shape[1] == 2:
                    probabilities = proba[:, 1]  # Probability of positive class
                elif proba.shape[1] == 1:
                    probabilities = proba[:, 0]
                else:
                    probabilities = None
            except:
                probabilities = None
        
        results = {
            'model': model,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'predictions': predictions,
            'probabilities': probabilities,
            'y_test': y_test,
            'X_test': X_test
        }
        
        if probabilities is not None:
            try:
                results['roc_auc'] = roc_auc_score(y_test, probabilities)
            except:
                results['roc_auc'] = None
        else:
            results['roc_auc'] = None
            
        return results
    except Exception as e:
        st.error(f"Error evaluating model {model_name}: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ==================== Page Functions ====================

def page_overview(df):
    """Overview page with project introduction and dataset summary"""
    st.markdown('<div class="main-header">Thyroid Cancer Recurrence Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">An AI-powered clinical decision support tool to assess recurrence risk in thyroid cancer patients.</div>', unsafe_allow_html=True)
    
    # Dataset Summary Cards
    st.markdown("## 📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🧪 Total Patients", len(df))
    with col2:
        if 'Recurred' in df.columns:
            num_recurred = df['Recurred'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0).sum()
            st.metric("⚕️ Recurrence Cases", num_recurred)
        else:
            st.metric("⚕️ Recurrence Cases", "N/A")
    with col3:
        st.metric("🔬 Features", len(df.columns) - 1)
    with col4:
        if 'Recurred' in df.columns:
            rate = df['Recurred'].apply(lambda x: 1 if x == 'Yes' or x == 1 else 0).mean() * 100
            st.metric("📈 Recurrence Rate", f"{rate:.1f}%")
        else:
            st.metric("📈 Recurrence Rate", "N/A")
    
    st.markdown("---")
    
    # Project Description
    st.markdown("## 📋 Project Objective")
    st.write("""
    This application helps clinicians and researchers:
    - Analyze thyroid cancer patient data
    - Understand key recurrence risk factors
    - Compare multiple machine learning models
    - Make individualized recurrence risk predictions
    """)
    
    # Dataset Preview
    st.markdown("## 🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Feature Information
    st.markdown("## 🧬 Feature Information")
    feature_info = pd.DataFrame({
        'Feature': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing Values': df.isnull().sum().values
    })
    st.dataframe(feature_info, use_container_width=True)

def page_eda(df):
    """Exploratory Data Analysis page"""
    st.markdown('# 🔍 Exploratory Data Analysis')
    st.markdown("### Interactive Data Exploration and Visualization")
    
    if df.empty:
        st.warning("No data available for EDA.")
        return
    
    # Sidebar filters
    st.sidebar.markdown("## 🎛️ Filter Controls")
    
    # Convert Recurred for analysis
    df_analysis = df.copy()
    if 'Recurred' in df_analysis.columns:
        df_analysis['Recurred_Numeric'] = df_analysis['Recurred'].apply(
            lambda x: 1 if x == 'Yes' or x == 1 else 0
        )
    
    st.markdown("---")
    
    # 1. Target Distribution
    st.markdown("## 📊 Target Variable Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Recurred' in df_analysis.columns:
            recurred_counts = df_analysis['Recurred'].value_counts()
            fig_pie = px.pie(
                values=recurred_counts.values,
                names=recurred_counts.index,
                title="Recurrence Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'Recurred' in df_analysis.columns:
            recurred_counts = df_analysis['Recurred'].value_counts()
            fig_bar = px.bar(
                x=recurred_counts.index,
                y=recurred_counts.values,
                title="Recurrence Count",
                labels={'x': 'Recurred', 'y': 'Count'},
                color=recurred_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # 2. Age Distribution
    st.markdown("## 👥 Age Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Age' in df_analysis.columns:
            fig_age = px.histogram(
                df_analysis,
                x='Age',
                nbins=20,
                title="Age Distribution",
                color_discrete_sequence=['#667eea'],
                marginal="box"
            )
            fig_age.update_layout(showlegend=False)
            st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        if 'Age' in df_analysis.columns and 'Recurred' in df_analysis.columns:
            fig_age_recur = px.box(
                df_analysis,
                x='Recurred',
                y='Age',
                title="Age vs Recurrence",
                color='Recurred',
                color_discrete_map={'Yes': '#ff4757', 'No': '#2ed573'}
            )
            st.plotly_chart(fig_age_recur, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Categorical Features Analysis
    st.markdown("## 📈 Categorical Features Analysis")
    
    categorical_cols = df_analysis.select_dtypes(include=['object']).columns.tolist()
    if 'Recurred' in categorical_cols:
        categorical_cols.remove('Recurred')
    
    if categorical_cols:
        selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count plot
            cat_counts = df_analysis[selected_cat].value_counts()
            fig_cat = px.bar(
                x=cat_counts.index,
                y=cat_counts.values,
                title=f"{selected_cat} Distribution",
                labels={'x': selected_cat, 'y': 'Count'},
                color=cat_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Stacked bar chart with Recurred
            if 'Recurred' in df_analysis.columns:
                cross_tab = pd.crosstab(df_analysis[selected_cat], df_analysis['Recurred'])
                fig_stacked = px.bar(
                    cross_tab,
                    title=f"{selected_cat} vs Recurrence",
                    labels={'value': 'Count', 'index': selected_cat},
                    color_discrete_map={'Yes': '#ff4757', 'No': '#2ed573'}
                )
                fig_stacked.update_layout(barmode='stack')
                st.plotly_chart(fig_stacked, use_container_width=True)
    
    st.markdown("---")
    
    # 4. Correlation Analysis
    st.markdown("## 🔗 Correlation Analysis")
    
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr_matrix = df_analysis[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto',
            labels=dict(color="Correlation")
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Display correlation with target
        if 'Recurred_Numeric' in numeric_cols:
            st.markdown("### Correlation with Recurrence")
            target_corr = corr_matrix['Recurred_Numeric'].sort_values(ascending=False)
            target_corr = target_corr[target_corr.index != 'Recurred_Numeric']
            
            fig_target_corr = px.bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation='h',
                title="Feature Correlation with Recurrence",
                labels={'x': 'Correlation', 'y': 'Feature'},
                color=target_corr.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_target_corr, use_container_width=True)
    
    st.markdown("---")
    
    # 5. Risk Level Analysis
    if 'Risk' in df_analysis.columns:
        st.markdown("## ⚠️ Risk Level Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = df_analysis['Risk'].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col2:
            if 'Recurred' in df_analysis.columns:
                risk_recur = pd.crosstab(df_analysis['Risk'], df_analysis['Recurred'])
                fig_risk_recur = px.bar(
                    risk_recur,
                    title="Risk Level vs Recurrence",
                    labels={'value': 'Count'},
                    color_discrete_map={'Yes': '#ff4757', 'No': '#2ed573'}
                )
                fig_risk_recur.update_layout(barmode='group')
                st.plotly_chart(fig_risk_recur, use_container_width=True)

def page_model_performance():
    """Model Performance Comparison Page"""
    st.markdown('# 📈 Model Performance Comparison')
    st.markdown("### Evaluating 5 Machine Learning Models (3 Trained + 2 Pre-trained)")
    
    # Load data
    df = load_data()
    if df.empty:
        st.error("Unable to load dataset. Please check if Thyroid_Diff.csv exists.")
        return
    
    # Load pre-trained models
    st.markdown("## 🔄 Loading Pre-trained Models")
    pre_trained_models, model_info = load_pre_trained_models()
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📌 Model Status")
        model_status_rows = []
        
        # Trained models (to be trained on this page)
        model_status_rows.append({"Model": "Random Forest", "Type": "Trained", "Status": "⏳ Pending Training"})
        model_status_rows.append({"Model": "XGBoost", "Type": "Trained", "Status": "⏳ Pending Training"})
        model_status_rows.append({"Model": "Gradient Boosting", "Type": "Trained", "Status": "⏳ Pending Training"})
        
        # Pre-trained models
        for name, info in model_info.items():
            status = "✅ Loaded" if info["loaded"] else "❌ Not Available"
            model_status_rows.append({"Model": name, "Type": "Pre-trained", "Status": status})
        
        model_status_df = pd.DataFrame(model_status_rows)
        st.dataframe(model_status_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### ⚙️ Evaluation Settings")
        st.write("""
        - Train 3 models on current dataset
        - Evaluate pre-trained models on same dataset (if compatible)
        - Compare metrics:
            - Accuracy
            - Precision
            - Recall
            - F1-Score
            - ROC-AUC
        """)
    
    st.markdown("---")
    
    st.markdown("## 🧪 Run Model Training & Evaluation")
    st.info("Click the button below to train the 3 models and evaluate all available models on the dataset.")
    
    if st.button("🚀 Train & Evaluate All Models", type="primary"):
        with st.spinner("Training and evaluating models..."):
            try:
                # Preprocess data
                X, y, le_dict, feature_cols = preprocess_data(df)
                if X.empty or y is None:
                    st.error("Data preprocessing failed. Please check the dataset.")
                    return
                
                # Ensure X is a DataFrame
                if isinstance(X, np.ndarray):
                    X = pd.DataFrame(X, columns=feature_cols)
                
                # Convert y to writable numpy array for train_test_split
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_array = np.array(y.values if hasattr(y, 'values') else y, copy=True)
                else:
                    y_array = np.array(y, copy=True)
                
                # Ensure y is contiguous and writable
                y_array = np.ascontiguousarray(y_array)
                
                # Split data once for trained models
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_array, test_size=0.2, random_state=42, stratify=y_array
                )
                
                results = {}
                all_models = {}
                
                # 1. Train the 3 algorithms
                st.markdown("### 🌲 Training 3 ML Algorithms")
                trained_models, trained_results = train_models(X_train, y_train, X_test, y_test)
                
                # Add trained models to results
                for model_name, result in trained_results.items():
                    results[model_name] = result
                    all_models[model_name] = result['model']
                
                # 2. Evaluate pre-trained models on a clinically relevant feature subset
                if pre_trained_models:
                    st.markdown("### 📦 Evaluating Pre-trained Models")
                    
                    if 'Recurred' not in df.columns:
                        st.warning("Target column 'Recurred' is missing; cannot evaluate pre-trained models.")
                    else:
                        # 👉 Give PyCaret the same kind of data it saw during training:
                        # all original columns EXCEPT the target.
                        X_pre = df.drop(columns=['Recurred']).copy()

                        # Prepare y for metrics (map Yes/No to 1/0 if needed)
                        y_series_pre = df['Recurred']
                        if y_series_pre.dtype == 'O':
                            mapping = {
                                'No': 0, 'no': 0, 'NO': 0,
                                'Yes': 1, 'yes': 1, 'YES': 1
                            }
                            y_array_pre = np.array(
                                [mapping.get(v, 1 if str(v).lower() == 'yes' else 0) for v in y_series_pre],
                                dtype=int
                            )
                        else:
                            y_array_pre = np.array(y_series_pre.values, copy=True)

                        y_array_pre = np.ascontiguousarray(y_array_pre)

                        # Split data for evaluation
                        X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(
                            X_pre, y_array_pre, test_size=0.2, random_state=42, stratify=y_array_pre
                        )

                        # Now evaluate each pre-trained model
                        for model_name, model in pre_trained_models.items():
                            result = evaluate_model_on_data(model, X_test_pre, y_test_pre, model_name)
                            if result:
                                results[model_name] = result
                                all_models[model_name] = model
                
                if results:
                    st.session_state['models'] = all_models
                    st.session_state['results'] = results
                    st.session_state['le_dict'] = le_dict
                    st.session_state['feature_cols'] = feature_cols
                    st.session_state['X'] = X
                    st.session_state['y'] = y_array
                    
                    # Determine best model
                    if results:
                        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
                        st.session_state['best_model_name'] = best_model_name
                        st.session_state['best_model'] = all_models[best_model_name]
                        st.success(
                            f"✅ All models evaluated successfully! Best model: "
                            f"**{best_model_name}** (F1-Score: {results[best_model_name]['f1']*100:.2f}%)"
                        )
            
            except Exception as e:
                st.error(f"Error during model training/evaluation: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    if 'results' not in st.session_state:
        st.info("👆 Click the button above to train and evaluate all models and view performance metrics.")
        
        # Show model info even if not evaluated
        st.markdown("## 📋 Available Models")
        model_list = [
            {'Model Name': 'Random Forest', 'Type': 'Trained', 'Status': '⏳ Not trained yet'},
            {'Model Name': 'XGBoost', 'Type': 'Trained', 'Status': '⏳ Not trained yet'},
            {'Model Name': 'Gradient Boosting', 'Type': 'Trained', 'Status': '⏳ Not trained yet'},
        ]
        
        # Add pre-trained models info
        for model_name, info in model_info.items():
            status = '✅ Loaded' if info['loaded'] else '❌ Failed'
            model_list.append({
                'Model Name': model_name,
                'Type': 'Pre-trained',
                'Status': status
            })
        
        info_df = pd.DataFrame(model_list)
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        return
    
    # If we have results, display performance comparison
    results = st.session_state['results']
    
    st.markdown("---")
    
    # Performance Metrics Table
    st.markdown("## 📊 Performance Metrics")
    
    metrics_data = []
    for model_name, res in results.items():
        roc_auc = res.get('roc_auc', None)
        metrics_data.append({
            'Model': model_name,
            'Accuracy': f"{res['accuracy']*100:.2f}%",
            'Precision': f"{res['precision']*100:.2f}%",
            'Recall': f"{res['recall']*100:.2f}%",
            'F1-Score': f"{res['f1']*100:.2f}%",
            'ROC-AUC': f"{roc_auc*100:.2f}%" if roc_auc is not None else "N/A"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Best Model Highlight
    best_model_name = st.session_state.get('best_model_name', list(results.keys())[0])
    if best_model_name in results:
        st.success(f"🏆 Best Performing Model: **{best_model_name}** (F1-Score: {results[best_model_name]['f1']*100:.2f}%)")
    
    st.markdown("---")
    
    # Visual Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Accuracy Comparison")
        fig_acc = px.bar(
            metrics_df,
            x='Model',
            y='Accuracy',
            text='Accuracy',
            title="Model Accuracy Comparison"
        )
        fig_acc.update_traces(textposition='outside')
        fig_acc.update_layout(yaxis_tickformat='.1f')
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 F1-Score Comparison")
        fig_f1 = px.bar(
            metrics_df,
            x='Model',
            y='F1-Score',
            text='F1-Score',
            title="Model F1-Score Comparison"
        )
        fig_f1.update_traces(textposition='outside')
        fig_f1.update_layout(yaxis_tickformat='.1f')
        st.plotly_chart(fig_f1, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.markdown("## 🔍 Confusion Matrices")
    
    num_models = len(results)
    cols = st.columns(num_models)
    
    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx]:
            cm = confusion_matrix(result['y_test'], result['predictions'])
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual"),
                x=['No Recurrence', 'Recurrence'],
                y=['No Recurrence', 'Recurrence'],
                color_continuous_scale='Blues',
                title=f"{model_name}",
                text_auto=True
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("---")
    
    # ROC Curves
    if any(results[m]['probabilities'] is not None for m in results.keys()):
        st.markdown("## 📈 ROC Curves")
        
        fig_roc = go.Figure()
        
        for model_name, result in results.items():
            if result['probabilities'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(result['y_test'], result['probabilities'])
                    auc = result.get('roc_auc', roc_auc_score(result['y_test'], result['probabilities']))
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name=f'{model_name} (AUC = {auc:.3f})',
                        line=dict(width=2)
                    ))
                except Exception as e:
                    st.warning(f"Could not plot ROC curve for {model_name}: {e}")
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(dash='dash', width=2)
        ))
        
        fig_roc.update_layout(
            title="ROC Curves for All Models",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importances (for tree-based models)
    st.markdown("## 🌿 Feature Importances (Tree-based Models)")
    
    feature_cols = st.session_state.get('feature_cols', None)
    if not feature_cols:
        st.info("Feature names are not available. Please rerun training to view feature importances.")
        return
    
    model_selection = st.selectbox(
        "Select a model to view feature importances",
        [m for m in results.keys() if m in ['Random Forest', 'XGBoost', 'Gradient Boosting']]
    )
    
    if model_selection:
        model = st.session_state['models'][model_selection]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Align importances with feature names
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                # Fall back to stored feature_cols
                feature_names = feature_cols
            
            # If lengths mismatch, handle gracefully
            if len(feature_names) != len(importances):
                if len(importances) < len(feature_names):
                    feature_names = feature_names[:len(importances)]
                else:
                    # Pad feature names if fewer than importances
                    feature_names = feature_names + [
                        f'Feature_{i}' for i in range(len(feature_names), len(importances))
                    ]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False).head(15)
            
            fig_imp = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Top 15 Important Features - {model_selection}",
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning(f"Feature importances not available for {model_selection}.")

def page_prediction():
    """Prediction Page"""
    st.markdown('# 🔮 Recurrence Risk Prediction')
    st.markdown("### Use the best-performing model to predict recurrence risk for a new patient.")
    
    df = load_data()
    if df.empty:
        st.error("Unable to load dataset. Please check if Thyroid_Diff.csv exists.")
        return
    
    # Ensure we have a best model
    best_model = st.session_state.get('best_model', None)
    best_model_name = st.session_state.get('best_model_name', None)
    
    if best_model is None or best_model_name is None:
        st.warning("Please go to the **Model Performance** page and run model training & evaluation first.")
        return
    
    st.info(f"Using **{best_model_name}** as the prediction model.")
    
    feature_cols = st.session_state.get('feature_cols', None)
    if feature_cols is None:
        st.error("Feature columns not found in session. Please rerun the model training process.")
        return
    
    st.markdown("## 🧾 Patient Information Input")
    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = {}
        
        for i, feature in enumerate(feature_cols):
            with cols[i % 3]:
                if df[feature].dtype in ['int64', 'float64']:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    user_input[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )
                else:
                    unique_vals = df[feature].dropna().unique().tolist()
                    if len(unique_vals) == 0:
                        user_input[feature] = st.text_input(feature, "")
                    else:
                        user_input[feature] = st.selectbox(feature, unique_vals)
        
        submitted = st.form_submit_button("Predict Recurrence Risk")
    
    if submitted:
        input_df = pd.DataFrame([user_input])
        
        # Prepare data similar to training preprocessing
        # Encode categoricals using the training label encoders if available
        le_dict = st.session_state.get('le_dict', {})
        for col, le in le_dict.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    # Handle unseen labels by mapping to the closest known label or a default
                    known_classes = list(le.classes_)
                    input_df[col] = input_df[col].apply(
                        lambda x: known_classes[0] if x not in known_classes else x
                    )
                    input_df[col] = le.transform(input_df[col].astype(str))
        
        # Align columns with training features
        X_train = st.session_state.get('X', None)
        if X_train is None:
            st.error("Training features not found in session. Please rerun the model training process.")
            return
        
        train_cols = list(X_train.columns)
        input_df = input_df.reindex(columns=train_cols, fill_value=0)
        
        # Predict
        try:
            proba = best_model.predict_proba(input_df)[0][1]
        except Exception:
            try:
                proba = best_model.predict_proba(input_df)[0]
            except Exception:
                proba = None
        
        pred = best_model.predict(input_df)[0]
        
        # Display results
        st.markdown("## 🧠 Prediction Result")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_label = "High Recurrence Risk" if pred == 1 else "Low Recurrence Risk"
            risk_color = "#dc2626" if pred == 1 else "#16a34a"
            confidence = proba if proba is not None else 0.5
            
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="stat-label">Predicted Risk</div>
                    <div class="stat-value" style="color:{risk_color};">{risk_label}</div>
                    <div class="stat-caption">
                        Model: <strong>{best_model_name}</strong><br>
                        Confidence: <strong>{confidence*100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown("### 🔍 Probability Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={'text': "Recurrence Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 30], 'color': "#bbf7d0"},
                        {'range': [30, 70], 'color': "#fef9c3"},
                        {'range': [70, 100], 'color': "#fecaca"},
                    ],
                    'bar': {'color': risk_color}
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("## 📌 Interpretation")
        if pred == 1:
            st.warning("""
            The model predicts **high risk of recurrence**. This does **not** mean recurrence is certain,
            but suggests closer follow-up and more intensive management **may** be warranted.
            """)
        else:
            st.success("""
            The model predicts **low risk of recurrence**. Regular follow-up is still important, but
            the immediate risk appears lower based on the model.
            """)
        
        st.caption("⚠️ This tool is for research and educational purposes only and should not replace clinical judgment.")

# ==================== Main App ====================

def main():
    df = load_data()
    if df.empty:
        st.error("Failed to load dataset. Please ensure Thyroid_Diff.csv is available.")
        return
    
    st.sidebar.markdown('<div class="sidebar-title">Thyroid Cancer AI</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "EDA", "Model Performance", "Prediction"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("Developed for **ARTI-404 Web Development and Cloud Computing**")
    
    if page == "Overview":
        page_overview(df)
    elif page == "EDA":
        page_eda(df)
    elif page == "Model Performance":
        page_model_performance()
    elif page == "Prediction":
        page_prediction()

if __name__ == "__main__":
    main()
