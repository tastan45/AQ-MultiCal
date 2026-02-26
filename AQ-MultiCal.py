import streamlit as st
import pandas as pd
import time
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

# NEW IMPORTS FOR ADDITIONAL MODELS
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# --- NEW IMPORTS FOR BAYESIAN OPTIMIZATION ---
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    _SKOPT_AVAILABLE = True
except ImportError:
    _SKOPT_AVAILABLE = False
# --- END OF NEW IMPORTS ---

# Scipy is imported only for KDE plotting, otherwise it won't throw an error
try:
    import scipy.stats as stats
    _SCIPY_AVAILABLE = True
    from scipy.stats import loguniform, uniform, randint
except ImportError:
    _SCIPY_AVAILABLE = False

# --- NEW IMPORTS FOR SEABORN PLOTTING ---
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False
# --- END OF NEW IMPORTS ---


# --- EXISTING NEW MODEL LIBRARIES FROM YOUR ORIGINAL CODE ---
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
# --- End of library additions ---


try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        non_zero_indices = y_true != 0
        if not np.any(non_zero_indices):
            return np.nan
        y_true_filtered = y_true[non_zero_indices]
        y_pred_filtered = y_pred[non_zero_indices]
        return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100

# V30.2
POLLUTANT_DISPLAY_UNITS = {
    'CO2': 'ppm',
    'PM1': 'µg/m³',
    'PM25': 'µg/m³',
    'PM10': 'µg/m³',
    'TEMPERATURE': '°C',
    'HUMIDITY': '%RH'
}

# --- Application Interface Settings ---
st.set_page_config(page_title="Hyperparameter Optimization for Air Quality Sensor Calibration", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS Styles (For Professional Look) ---
st.title("⚙️ Hyperparameter Optimization for Air Quality Sensor Calibration")

# --- NEW: COMBINED PARAMETER POOL FOR ALL OPTIMIZATION TYPES ---
COMMON_PARAM_GRID = {
    "Random Forest": {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, 30, None], # None means unlimited depth
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0], 
        'bootstrap': [True, False]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'min_samples_leaf': [1, 3, 5]
    },
      "Support Vector Regression (SVR)": {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'epsilon': [0.01, 0.05, 0.1, 0.2, 0.5]
    },
    "k-Nearest Neighbors (kNN)": {
        'n_neighbors': [3, 5, 7, 11, 15, 20, 30,40],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': [1, 2]
    },
    "Linear Regression": {
        'fit_intercept': [True, False]
    },
    "Decision Tree": {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None, 1.0] # None means all features
    },
    "AdaBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'loss': ['linear', 'square', 'exponential'],
        'base_estimator__max_depth': [1, 2, 3] # AdaBoost's base estimator (Decision Tree) depth
    },
    "SGD": {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet', None],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.01, 0.1, 0.5] # Initial learning rate, depends on learning_rate
    },
    "Ridge Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False]
    },
    "Lasso Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False]
    },
    "ElasticNet Regression": {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio': [0.1, 0.5, 0.9],
        'fit_intercept': [True, False]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # Tuple of (neurons_in_layer1, ...)
        'activation': ['relu', 'tanh', 'logistic'], # Activation function for the hidden layer
        'solver': ['adam', 'sgd', 'lbfgs'], # Algorithm for weight optimization
        'alpha': [0.0001, 0.001, 0.01], # L2 regularization term
        'learning_rate_init': [0.001, 0.01, 0.1] # Initial learning rate for 'adam' or 'sgd'
    },
    "XGBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    "LightGBM": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'max_depth': [-1, 5, 7], # -1 means no limit
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    "CatBoost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5],
    }
}

# --- RANDOMIZEDSEARCHCV PARAMETER DISTRIBUTIONS ---
# Using scipy.stats distributions for more efficient search
RANDOM_PARAM_SPACES = {
    "Random Forest": {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 20, 30, None], 
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
        'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0],
        'bootstrap': [True, False]
    },
    "Gradient Boosting": {
        'n_estimators': randint(50, 300),
        'learning_rate': loguniform(0.01, 0.2),
        'max_depth': randint(3, 7),
        'subsample': uniform(0.6, 0.4), # Start at 0.6, search range 0.4
        'min_samples_leaf': randint(1, 5)
    },
    "Support Vector Regression (SVR)": {
        'C': loguniform(0.1, 1000),
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'epsilon': uniform(0.01, 0.49) # Start at 0.01, search range 0.49
    },
    "k-Nearest Neighbors (kNN)": {
        'n_neighbors': randint(3, 40),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'p': randint(1, 2)
    },
    "Linear Regression": {
        'fit_intercept': [True, False]
    },
    "Decision Tree": {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
        'max_features': ['sqrt', 'log2', None, 1.0]
    },
    "AdaBoost": {
        'n_estimators': randint(50, 200),
        'learning_rate': loguniform(0.01, 0.2),
        'loss': ['linear', 'square', 'exponential'],
        'base_estimator__max_depth': randint(1, 3) # Corrected parameter
    },
    "SGD": {
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet', None],
        'alpha': loguniform(1e-4, 1e-2),
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': uniform(0.01, 0.49)
    },
    "Ridge Regression": {
        'alpha': loguniform(0.01, 100.0),
        'fit_intercept': [True, False]
    },
    "Lasso Regression": {
        'alpha': loguniform(0.01, 100.0),
        'fit_intercept': [True, False]
    },
    "ElasticNet Regression": {
        'alpha': loguniform(0.01, 100.0),
        'l1_ratio': uniform(0.1, 0.8),
        'fit_intercept': [True, False]
    },
    "MLP": {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd', 'lbfgs'],
        'alpha': loguniform(1e-4, 1e-2),
        'learning_rate_init': loguniform(1e-3, 0.1)
    },
    "XGBoost": {
        'n_estimators': randint(50, 200),
        'learning_rate': loguniform(0.01, 0.1),
        'max_depth': randint(3, 7),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    },
    "LightGBM": {
        'n_estimators': randint(50, 200),
        'learning_rate': loguniform(0.01, 0.1),
        'num_leaves': randint(20, 40),
        'max_depth': [-1, 5, 7], # -1 means no limit
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4)
    },
    "CatBoost": {
        'n_estimators': randint(50, 200),
        'learning_rate': loguniform(0.01, 0.1),
        'depth': randint(4, 8),
        'l2_leaf_reg': randint(1, 5),
    }
}


# --- NEW: BAYESIAN OPTIMIZATION PARAMETER SPACES (for skopt) ---
BAYES_PARAM_SPACES = {
    "Support Vector Regression (SVR)": {
        'C': Real(1e-2, 1e3, "log-uniform", name='C'),
        'gamma': Categorical([0.001, 0.01, 0.1, 1, 'scale', 'auto'], name='gamma'), 
        'kernel': Categorical(['rbf', 'linear', 'poly', 'sigmoid'], name='kernel'),
        'epsilon': Real(1e-3, 0.5, "uniform", name='epsilon')
    },
    "Random Forest": {
        'n_estimators': Integer(50, 300, name='n_estimators'),
        'max_depth': Categorical([5, 10, 20, 30, None], name='max_depth'), 
        'min_samples_split': Integer(2, 10, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 4, name='min_samples_leaf'),
        'max_features': Categorical(['sqrt', 'log2', 0.6, 0.8, 1.0], name='max_features'),
        'bootstrap': Categorical([True, False], name='bootstrap')
    },
    "Gradient Boosting": {
        'n_estimators': Integer(50, 300, name='n_estimators'),
        'learning_rate': Real(0.01, 0.2, 'log-uniform', name='learning_rate'),
        'max_depth': Integer(3, 7, name='max_depth'),
        'subsample': Real(0.6, 1.0, 'uniform', name='subsample'),
        'min_samples_leaf': Integer(1, 5, name='min_samples_leaf')
    },
    "XGBoost": {
        'n_estimators': Integer(50, 200, name='n_estimators'),
        'learning_rate': Real(0.01, 0.1, 'log-uniform', name='learning_rate'),
        'max_depth': Integer(3, 7, name='max_depth'),
        'subsample': Real(0.6, 1.0, 'uniform', name='subsample'),
        'colsample_bytree': Real(0.6, 1.0, 'uniform', name='colsample_bytree')
    },
    "LightGBM": {
        'n_estimators': Integer(50, 200, name='n_estimators'),
        'learning_rate': Real(0.01, 0.1, 'log-uniform', name='learning_rate'),
        'num_leaves': Integer(20, 40, name='num_leaves'),
        'max_depth': Integer(-1, 7, name='max_depth'), # -1 is a valid option for LightGBM
        'subsample': Real(0.6, 1.0, 'uniform', name='subsample'),
        'colsample_bytree': Real(0.6, 1.0, 'uniform', name='colsample_bytree')
    },
    "CatBoost": {
        'n_estimators': Integer(50, 200, name='n_estimators'),
        'learning_rate': Real(0.01, 0.1, 'log-uniform', name='learning_rate'),
        'depth': Integer(4, 8, name='depth'),
        'l2_leaf_reg': Integer(1, 5, name='l2_leaf_reg')
    },
    "AdaBoost": {
        'n_estimators': Integer(50, 200, name='n_estimators'),
        'learning_rate': Real(0.01, 0.2, 'log-uniform', name='learning_rate'),
        'loss': Categorical(['linear', 'square', 'exponential'], name='loss'),
        'base_estimator__max_depth': Integer(1, 3, name='base_estimator__max_depth') # Corrected parameter
    },
    "k-Nearest Neighbors (kNN)": {
        'n_neighbors': Integer(3, 40, name='n_neighbors'),
        'weights': Categorical(['uniform', 'distance'], name='weights'),
        'metric': Categorical(['euclidean', 'manhattan'], name='metric'),
        'p': Integer(1, 2, name='p')
    },
    "Linear Regression": {
        'fit_intercept': Categorical([True, False], name='fit_intercept')
    },
    "Decision Tree": {
        'max_depth': Categorical([5, 10, 20, 30, None], name='max_depth'),
        'min_samples_split': Integer(2, 10, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 4, name='min_samples_leaf'),
        'max_features': Categorical(['sqrt', 'log2', None, 1.0], name='max_features')
    },
    "SGD": {
        'loss': Categorical(['squared_error', 'huber', 'epsilon_insensitive'], name='loss'),
        'penalty': Categorical(['l2', 'l1', 'elasticnet', None], name='penalty'),
        'alpha': Real(1e-4, 1e-2, 'log-uniform', name='alpha'),
        'learning_rate': Categorical(['constant', 'optimal', 'invscaling', 'adaptive'], name='learning_rate'),
        'eta0': Real(0.01, 0.5, 'uniform', name='eta0')
    },
    "Ridge Regression": {
        'alpha': Real(0.01, 100.0, 'log-uniform', name='alpha'),
        'fit_intercept': Categorical([True, False], name='fit_intercept')
    },
    "Lasso Regression": {
        'alpha': Real(0.01, 100.0, 'log-uniform', name='alpha'),
        'fit_intercept': Categorical([True, False], name='fit_intercept')
    },
    "ElasticNet Regression": {
        'alpha': Real(0.01, 100.0, 'log-uniform', name='alpha'),
        'l1_ratio': Real(0.1, 0.9, 'uniform', name='l1_ratio'),
        'fit_intercept': Categorical([True, False], name='fit_intercept')
    },
    "MLP": {
        'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50)], name='hidden_layer_sizes'),
        'activation': Categorical(['relu', 'tanh', 'logistic'], name='activation'),
        'solver': Categorical(['adam', 'sgd', 'lbfgs'], name='solver'),
        'alpha': Real(1e-4, 1e-2, 'log-uniform', name='alpha'),
        'learning_rate_init': Real(1e-3, 0.1, 'log-uniform', name='learning_rate_init')
    }
}


# --- TOP 3 PARAMETERS FOR AUTOMATIC OPTIMIZATION ---
AUTO_OPTIMIZE_PARAMS = {
    "Random Forest": ['n_estimators', 'max_depth','min_samples_split'],
    "Gradient Boosting": ['n_estimators', 'learning_rate','max_depth'],
    "Support Vector Regression (SVR)": ['C', 'gamma','kernel'],
    "k-Nearest Neighbors (kNN)": ['n_neighbors', 'weights','metric'],
    "Linear Regression": ['fit_intercept'],
    "Decision Tree": ['max_depth', 'min_samples_split','min_samples_leaf'],
    "AdaBoost": ['n_estimators', 'learning_rate','loss'], # Corrected parameter
    "SGD": ['alpha', 'learning_rate','penalty'],
    "Ridge Regression": ['alpha', 'fit_intercept'],
    "Lasso Regression": ['alpha', 'fit_intercept'],
    "ElasticNet Regression": ['alpha', 'l1_ratio','fit_intercept'],
    "MLP": ['hidden_layer_sizes', 'activation','solver'],
    # --- AUTO-OPTIMIZE PARAMS FOR NEWLY ADDED MODELS ---
    "XGBoost": ['n_estimators', 'learning_rate','max_depth'],
    "LightGBM": ['n_estimators', 'learning_rate','num_leaves'],
    "CatBoost": ['n_estimators', 'learning_rate','depth']
}

# --- Model Instances (using default parameters, will be overridden if optimized) ---
MODELS = {
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "k-Nearest Neighbors (kNN)": KNeighborsRegressor(n_jobs=-1),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3, random_state=42), random_state=42),
    "SGD": SGDRegressor(random_state=42),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "ElasticNet Regression": ElasticNet(random_state=42),
    "MLP": MLPRegressor(random_state=42, max_iter=1000),
    "Support Vector Regression (SVR)": SVR(),
    # --- INSTANCES FOR NEWLY ADDED MODELS ---
    "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1),
    "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
}

PLOTLY_CONFIG = {'toImageButtonOptions': {'format': 'png', 'filename': 'newplot', 'height': 800, 'width': 1200, 'scale': 5}}

# --- DEFAULT PLOT STYLES (New Global Variable with Hex Colors and Float Widths) ---
DEFAULT_PLOT_STYLES = {
    "general": {
        "template": "simple_white",
        "font_family": "Arial",
        "font_color": "#000000",
        "plot_title_font_size": 20,
        "axis_title_font_size": 14,
        "axis_title_font_color": "#333333",
        "axis_title_font_family": "Arial",
        "axis_tick_font_size": 12,
        "axis_tick_font_color": "#000000",
        "axis_tick_font_family": "Arial"
    },
    "time_series": {
        "raw_color": "#228B22",
        "raw_width": 1.0,
        "raw_style": "solid", "raw_opacity": 0.5,
        "calibrated_color": "#FF7F0E",
        "calibrated_width": 2.0,
        "calibrated_style": "solid", "calibrated_opacity": 1.0,
        "reference_color": "#000000",
        "reference_width": 3.0,
        "reference_style": "dash",
        "title": None,
        "xaxis_title": "Time",
        "yaxis_title": None,
        "legend_bgcolor": "#FFFFFF"
    },
    "scatter_plot": {
        "marker_color": "#228B22",
        "marker_size": 5, "marker_opacity": 0.5,
        "trendline_color": "#DC143C",
        "trendline_width": 2.5,
        "trendline_style": "solid",
        "title": None,
        "xaxis_title": None,
        "yaxis_title": None,
    },
    "residuals_plot": {
        "marker_color": "#228B22",
        "marker_size": 5, "marker_opacity": 0.5,
        "zeroline_color": "#646464",
        "zeroline_width": 2.0,
        "zeroline_style": "dash",
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Residuals"
    },
    "residuals_hist": {
        "bar_color": "#1F77B4",
        "bar_opacity": 0.7,
        "line_color": "#FF0000",
        "line_width": 2.0,
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Frequency"
    },
    "residuals_kde": {
        "line_color": "#9467BD",
        "fill_color": "#9467BD",
        "fill_opacity": 0.3,
        "title": None,
        "xaxis_title": None,
        "yaxis_title": "Density"
    }
}


# --- Helper function for consistent progress bar updates with ETA ---
def update_progress_bar_with_eta(progress_bar_obj, progress_percent, message):
    elapsed_time = time.time() - st.session_state.start_time
    
    remaining_str = "Estimating..."
    if progress_percent > 0:
        estimated_total = elapsed_time / (progress_percent / 100.0)
        remaining_time = estimated_total - elapsed_time
        
        if remaining_time < 0:
            remaining_str = "Completed."
        elif remaining_time < 60:
            remaining_str = f"~{remaining_time:.1f}s remaining"
        else:
            remaining_str = f"~{remaining_time / 60:.1f} min remaining"
    
    progress_bar_obj.progress(progress_percent, text=f"{message} (Elapsed: {elapsed_time:.1f}s | {remaining_str})")


# --- Main Functions (Function Definitions) ---
def calculate_all_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten() 
    
    non_zero_indices = y_true != 0
    if not np.any(non_zero_indices):
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "r2": np.nan}
    y_true_filtered = y_true[non_zero_indices]
    y_pred_filtered = y_pred[non_zero_indices]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


@st.cache_data(show_spinner="Merging data files...")
def merge_and_prepare_data(pollutant_data, temp_data, hum_data):
    df_p = pd.read_csv(pollutant_data); df_t = pd.read_csv(temp_data); df_h = pd.read_csv(hum_data)
    for df in [df_p, df_t, df_h]:
        df.columns = [col.lower().strip() for col in df.columns]
        df.rename(columns=lambda x: re.sub(r'timestamp', 'timestamp', x, re.IGNORECASE), inplace=True)
        df.rename(columns=lambda x: re.sub(r'temp(erature)?', 'temp', x), inplace=True)
        df.rename(columns=lambda x: re.sub(r'hum(idity)?', 'hum', x), inplace=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

    ref_col_search = [col for col in df_p.columns if col.startswith('reference_')]
    if not ref_col_search: raise ValueError("No reference column starting with 'reference_' found in the main pollutant file.")
    ref_col = ref_col_search[0]
    pollutant_name_raw = ref_col.split('reference_')[1]
    pollutant_name_upper = pollutant_name_raw.upper()

    display_unit = POLLUTANT_DISPLAY_UNITS.get(pollutant_name_upper, pollutant_name_upper)

    df_p.columns = [col.replace(f'_{pollutant_name_raw}', '_pollutant') if not col.startswith('reference') else 'reference_pollutant' for col in df_p.columns]
    
    df_merged = df_p.merge(df_t, on='timestamp', how='inner').merge(df_h, on='timestamp', how='inner')
    locations = sorted(list(set([col.split('_')[0] for col in df_merged.columns if not col.startswith('reference')])))
    
    all_locations_data = []
    for loc in locations:
        try:
            df_loc = pd.DataFrame({'reference_pollutant': df_merged['reference_pollutant'],'raw_pollutant': df_merged[f'{loc}_pollutant'], 'raw_temp': df_merged[f'{loc}_temp'],'raw_humidity': df_merged[f'{loc}_hum'], 'location': loc})
            df_loc.index = df_merged.index
            all_locations_data.append(df_loc)
        except KeyError: continue
    if not all_locations_data: raise ValueError("Location columns could not be matched. Ensure location prefixes (like 'kitchen_') in files are consistent.")
    
    df_final_long = pd.concat(all_locations_data).reset_index()
    return df_final_long.dropna(), pollutant_name_upper, display_unit


def plot_time_series(df_plot, pollutant_unit, location_name, model_name, display_unit, plot_config):
    # Dynamic Titles and Labels
    graph_title = plot_config["time_series"]["title"] or (f"{model_name} Model Time Series - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Overall' else f"{model_name} Model Time Series - Overall {pollutant_unit}")
    xaxis_title = plot_config["time_series"]["xaxis_title"]
    yaxis_title = plot_config["time_series"]["yaxis_title"] or (f"{location_name.capitalize()} {pollutant_unit} [{display_unit}]" if location_name != 'Overall' else f"Overall {pollutant_unit} [{display_unit}]")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['raw_pollutant'], mode='lines', name='Raw Sensor',
        line=dict(color=plot_config["time_series"]["raw_color"], width=plot_config["time_series"]["raw_width"], dash=plot_config["time_series"]["raw_style"]),
        opacity=plot_config["time_series"]["raw_opacity"]
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['calibrated_pollutant'], mode='lines', name='Calibrated (Model)',
        line=dict(color=plot_config["time_series"]["calibrated_color"], width=plot_config["time_series"]["calibrated_width"], dash=plot_config["time_series"]["calibrated_style"]),
        opacity=plot_config["time_series"]["calibrated_opacity"]
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['timestamp'], y=df_plot['reference_pollutant'], mode='lines', name='Reference Device',
        line=dict(color=plot_config["time_series"]["reference_color"], width=plot_config["time_series"]["reference_width"], dash=plot_config["time_series"]["reference_style"])
    ))
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        xaxis_title=xaxis_title, 
        yaxis_title=yaxis_title, 
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor=plot_config["time_series"]["legend_bgcolor"], bordercolor="Black", borderwidth=1),
        template=plot_config["general"]["template"], # Apply general template
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]), # Apply general font
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_scatter(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config, chart_key=None):
    r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred)
    N = len(y_test) # Data point count

    # Dynamic Titles and Labels
    graph_title = plot_config["scatter_plot"]["title"] or (f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit}" if location_name != 'Overall' else f"{model_name} Predictions for Overall {pollutant_unit}")
    xaxis_title = plot_config["scatter_plot"]["xaxis_title"] or f"Reference Values [{display_unit}]"
    yaxis_title = plot_config["scatter_plot"]["yaxis_title"] or (f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit} [{display_unit}]" if location_name != 'Overall' else f"{model_name} Predictions for Overall {pollutant_unit} [{display_unit}]")

    # --- DENSITY CALCULATION AND DATAFRAME CREATION START ---
    scatter_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

    range_min = min(scatter_df['y_test'].min(), scatter_df['y_pred'].min())
    range_max = max(scatter_df['y_test'].max(), scatter_df['y_pred'].max())
    bins = 50 # Number of bins for density, adjustable

    H, xedges, yedges = np.histogram2d(scatter_df['y_test'], scatter_df['y_pred'], bins=bins, 
                                        range=[[range_min, range_max], [range_min, range_max]])

    x_bin_indices = np.digitize(scatter_df['y_test'], xedges) - 1
    y_bin_indices = np.digitize(scatter_df['y_pred'], yedges) - 1

    x_bin_indices = np.clip(x_bin_indices, 0, bins - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, bins - 1)

    density = np.array([H[x_idx, y_idx] 
                        for x_idx, y_idx in zip(x_bin_indices, y_bin_indices)])
    scatter_df['density'] = density
    # --- DENSITY CALCULATION AND DATAFRAME CREATION END ---

    fig = px.scatter(
        scatter_df, 
        x='y_test', 
        y='y_pred', 
        color='density', # Color by density
        color_continuous_scale=px.colors.sequential.Jet, # Color scale for density (as in example)
        labels={'y_test': xaxis_title, 'y_pred': yaxis_title, 'density': 'Point Density'}, # Legend title for density
        opacity=plot_config["scatter_plot"]["marker_opacity"],
        trendline="ols",
        trendline_color_override=plot_config["scatter_plot"]["trendline_color"]
    )
    
    # --- ADD IDEAL 1:1 LINE START ---
    min_val_plot = min(y_test.min(), y_pred.min())
    max_val_plot = max(y_test.max(), y_pred.max())
    ideal_line_range = [min_val_plot, max_val_plot]

    fig.add_trace(go.Scatter(
        x=ideal_line_range,
        y=ideal_line_range,
        mode='lines',
        name='Ideal 1:1 Line',
        line=dict(color='black', dash='dash', width=2),
        showlegend=True
    ))
    # --- ADD LEGEND TO REGRESSION LINE END ---


    fig.update_traces(
        marker=dict(size=plot_config["scatter_plot"]["marker_size"]), # Color now comes from density, removed marker_color
        selector=dict(mode='markers')
    )
    fig.update_traces(
        line=dict(width=plot_config["scatter_plot"]["trendline_width"], dash=plot_config["scatter_plot"]["trendline_style"]),
        selector=dict(mode='lines')
    )
    
    # --- UPDATE ANNOTATION (add N value) ---
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", 
                       text=f"N = {N}<br>R² = {r2:.4f}<br>RMSE = {rmse:.2f}<br>MAE = {mae:.2f}", # N added
                       showarrow=False, align="left", bordercolor="black", borderwidth=1, bgcolor="#FFFFFF")
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        coloraxis_colorbar=dict( # Settings for density scale
            title="", # UPDATE: "Point Density" text removed
            len=1,   # Extend along Y-axis (normalized from 0 to 1)
            y=0.5,   # Vertically center
            x=1.08 # Adjust color bar position (to avoid being too far right)
        ),
        legend=dict( # Move legend to bottom right
            x=0.90,  # Close to right edge
            y=0.01,  # Close to bottom
            xanchor="right", # Align to right edge
            yanchor="bottom", # Align to bottom edge
            bgcolor="rgba(255,255,255,0.7)", # Background for readability
            bordercolor="Black",
            borderwidth=1
        ),
        # --- Axis tick label settings ---
        xaxis=dict(
            tickfont=dict( # Eksen tik etiketleri font ayarı (artık genel ayarları kullanıyor)
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            ),
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            )
        ),
        yaxis=dict(
            tickfont=dict( # Eksen tik etiketleri font ayarı (artık genel ayarları kullanıyor)
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            ),
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            )
        )
        # --- End of Axis tick label settings ---
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=chart_key)

def plot_scatter_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config, chart_key=None):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred)); mae = mean_absolute_error(y_test, y_pred)
    N = len(y_test)
    
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(
        x=y_test, y=y_pred, 
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'},
        ax=ax
    )
    
    min_val_plot = min(y_test.min(), y_pred.min())
    max_val_plot = max(y_test.max(), y_pred.max())
    ax.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], linestyle='--', color='black', label='Ideal 1:1 Line')

    ax.set_title(f"{model_name} Predictions for {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Reference Values [{display_unit}]")
    ax.set_ylabel(f"Predictions [{display_unit}]")
    ax.text(0.05, 0.95, f"N = {N}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    # Dynamic Titles and Labels
    graph_title = plot_config["residuals_plot"]["title"] or (f"{model_name} Model Residuals Plot - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Overall' else f"{model_name} Model Residuals Plot - Overall {pollutant_unit}")
    xaxis_title = plot_config["residuals_plot"]["xaxis_title"] or (f'{model_name} Model Predictions for {location_name.capitalize()} {pollutant_unit} [{display_unit}]' if location_name != 'Overall' else f'{model_name} Model Predictions for Overall {pollutant_unit} [{display_unit}]')
    yaxis_title = plot_config["residuals_plot"]["yaxis_title"]

    fig = px.scatter(
        x=y_pred, y=residuals, 
        labels={'x': xaxis_title, 'y': yaxis_title},
        opacity=plot_config["residuals_plot"]["marker_opacity"]
    )
    fig.update_traces(
        marker=dict(color=plot_config["residuals_plot"]["marker_color"], size=plot_config["residuals_plot"]["marker_size"])
    )
    fig.add_hline(
        y=0, 
        line_color=plot_config["residuals_plot"]["zeroline_color"], 
        line_dash=plot_config["residuals_plot"]["zeroline_style"],
        line_width=plot_config["residuals_plot"]["zeroline_width"]
    );
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"], # Apply general template
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]), # Apply general font
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_residuals_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return

    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.residplot(
        x=y_pred, y=y_test,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 2},
        ax=ax
    )

    ax.set_title(f"{model_name} Model Residuals Plot - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Predictions [{display_unit}]")
    ax.set_ylabel(f"Residuals [{display_unit}]")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# --- NEWLY ADDED GRAPH FUNCTIONS ---
def plot_residuals_histogram(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    graph_title = plot_config["residuals_hist"]["title"] or (f"{model_name} Model Residuals Distribution (Histogram) - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Overall' else f"{model_name} Model Residuals Distribution (Histogram) - Overall {pollutant_unit}")
    xaxis_title = plot_config["residuals_hist"]["xaxis_title"] or f"Residuals [{display_unit}]"
    yaxis_title = plot_config["residuals_hist"]["yaxis_title"]
    
    fig = px.histogram(
        x=residuals,
        nbins=50, # Default bin count, adjustable
        title=graph_title,
        labels={'x': xaxis_title, 'y': yaxis_title}
    )
    fig.update_traces(marker_color=plot_config["residuals_hist"]["bar_color"], opacity=plot_config["residuals_hist"]["bar_opacity"])
    
    # Mean and Median Lines
    fig.add_vline(x=np.mean(residuals), line_width=plot_config["residuals_hist"]["line_width"], line_dash="dash", line_color=plot_config["residuals_hist"]["line_color"], annotation_text=f"Mean: {np.mean(residuals):.2f}", annotation_position="top right")
    fig.add_vline(x=np.median(residuals), line_width=plot_config["residuals_hist"]["line_width"], line_dash="dot", line_color=plot_config["residuals_hist"]["line_color"], annotation_text=f"Median: {np.median(residuals):.2f}", annotation_position="bottom right")

    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
def plot_residuals_histogram_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    residuals = y_test - y_pred
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.histplot(residuals, kde=False, bins=50, ax=ax, color='skyblue')
    ax.axvline(x=np.mean(residuals), color='red', linestyle='--', label=f'Mean: {np.mean(residuals):.2f}')
    ax.axvline(x=np.median(residuals), color='red', linestyle=':', label=f'Median: {np.median(residuals):.2f}')
    
    ax.set_title(f"{model_name} Model Residuals Distribution (Histogram) - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Residuals [{display_unit}]")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_residuals_kde(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    residuals = y_test - y_pred
    
    graph_title = plot_config["residuals_kde"]["title"] or (f"{model_name} Model Residuals Density (KDE) - {location_name.capitalize()} {pollutant_unit}" if location_name != 'Overall' else f"{model_name} Model Residuals Density (KDE) - Overall {pollutant_unit}")
    xaxis_title = plot_config["residuals_kde"]["xaxis_title"] or f"Residuals [{display_unit}]"
    yaxis_title = plot_config["residuals_kde"]["yaxis_title"]
    
    # Using px.histogram to create a density plot
    fig = px.histogram(
        x=residuals,
        histnorm='density', # Important parameter for density plot
        nbins=50, # Desired number of bins, adjustable
        title=graph_title,
        labels={'x': xaxis_title, 'y': yaxis_title}
    )
    
    # Adjusting the appearance of histogram bars
    fig.update_traces(
        marker_color=plot_config["residuals_kde"]["fill_color"], # Color of histogram bars
        opacity=plot_config["residuals_kde"]["fill_opacity"], # Opacity of histogram bars
        selector=dict(type='histogram') # Select only histogram traces
    )
    
    # Optionally, we can add a KDE line to show the distribution more smoothly.
    # This may require the scipy library.
    if _SCIPY_AVAILABLE:
        try:
            kde = stats.gaussian_kde(residuals)
            x_vals = np.linspace(min(residuals), max(residuals), 500)
            y_vals = kde(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', 
                                     line=dict(color=plot_config["residuals_kde"]["line_color"], width=2), 
                                     name='KDE Curve', showlegend=False))
        except Exception as e:
            st.warning(f"Error plotting KDE curve: {e}. Please check residuals data.")
    else:
        st.info("Scipy library not installed. KDE curve cannot be plotted. Density distribution will be shown as histogram.")


    fig.update_layout(
        title=dict( # Başlık ayarı
            text=graph_title,
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=xaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=yaxis_title,
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_residuals_kde_seaborn(y_test, y_pred, pollutant_unit, location_name, model_name, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    residuals = y_test - y_pred
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.kdeplot(residuals, fill=True, color='purple', ax=ax, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', label='Zero Residuals')
    
    ax.set_title(f"{model_name} Model Residuals Density (KDE) - {location_name.capitalize()} {pollutant_unit}")
    ax.set_xlabel(f"Residuals [{display_unit}]")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_dataset_distributions(y_train, y_val, y_test, display_unit, plot_config):
    """
    Draws a box plot comparing the value distributions of
    Training, Validation, and Test sets.
    """
    # Create a long-form DataFrame suitable for Plotly Express
    df_train = pd.DataFrame({'Value': y_train, 'Set': 'Training'})
    df_val = pd.DataFrame({'Value': y_val, 'Set': 'Validation'})
    df_test = pd.DataFrame({'Value': y_test, 'Set': 'Test'})
    df_plot = pd.concat([df_train, df_val, df_test])

    fig = px.box(
        df_plot,
        x='Set',
        y='Value',
        color='Set',
        title="Train/Validation/Test Set Value Distributions",
        labels={'Value': f'Target Value [{display_unit}]', 'Set': 'Dataset'}
    )
    
    fig.update_layout(
        title=dict( # Başlık ayarı
            text="Train/Validation/Test Set Value Distributions",
            font=dict(
                size=plot_config["general"]["plot_title_font_size"], # Yeni eklenen başlık font boyutu
                family=plot_config["general"]["font_family"],
                color=plot_config["general"]["font_color"]
            )
        ),
        template=plot_config["general"]["template"],
        font=dict(family=plot_config["general"]["font_family"], color=plot_config["general"]["font_color"]),
        xaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text="Dataset",
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        ),
        yaxis=dict( # Eksen ayarları
            title=dict( # Eksen başlığı font ayarı
                text=f'Target Value [{display_unit}]',
                font=dict(
                    size=plot_config["general"]["axis_title_font_size"],
                    color=plot_config["general"]["axis_title_font_color"],
                    family=plot_config["general"]["axis_title_font_family"]
                )
            ),
            tickfont=dict( # Eksen tik etiketleri font ayarı
                size=plot_config["general"]["axis_tick_font_size"],
                color=plot_config["general"]["axis_tick_font_color"],
                family=plot_config["general"]["axis_tick_font_family"]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_dataset_distributions_seaborn(y_train, y_val, y_test, display_unit, plot_config):
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return
    
    df_train = pd.DataFrame({'Value': y_train, 'Set': 'Training'})
    df_val = pd.DataFrame({'Value': y_val, 'Set': 'Validation'})
    df_test = pd.DataFrame({'Value': y_test, 'Set': 'Test'})
    df_plot = pd.concat([df_train, df_val, df_test])

    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(x='Set', y='Value', data=df_plot, ax=ax, palette='Set2')
    
    ax.set_title("Train/Validation/Test Set Value Distributions")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(f"Target Value [{display_unit}]")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def plot_pair_plots_seaborn(df, title, plot_config):
    """
    Draws a pair plot using Seaborn for better customization,
    including scatter plots and KDE plots on the diagonal.
    """
    if not _SEABORN_AVAILABLE:
        st.warning("Seaborn or Matplotlib library not installed. Cannot plot with this function.")
        return

    # Create the pair plot
    sns.set_style("whitegrid", {"axes.facecolor": "#FFFFFF"})
    fig = sns.pairplot(df, diag_kind="kde", corner=True, plot_kws={'alpha': 0.6})

    # Set the title for the plot (this requires a bit of matplotlib magic)
    fig.fig.suptitle(title, y=1.02)
    
    # Customize the axes and labels if needed
    for ax in fig.axes.flatten():
        if ax:
            ax.xaxis.label.set_size(plot_config["general"]["axis_title_font_size"])
            ax.yaxis.label.set_size(plot_config["general"]["axis_title_font_size"])
            ax.tick_params(labelsize=plot_config["general"]["axis_tick_font_size"])
    
    st.pyplot(fig.figure)
    plt.close(fig.figure) # Close the plot to free memory

def plot_pair_plots(df, title, plot_config):
    # This function is now a fallback, but we will keep it as it's part of the original code.
    fig = px.scatter_matrix(df, dimensions=[col for col in df.columns if col != 'r2_score'], color='r2_score')
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(title=title)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def plot_parallel_coordinates(df, title, plot_config):
    # --- MODIFIED: ADDING CUSTOM COLORSCALE ---
    fig = px.parallel_coordinates(
        df, 
        dimensions=[col for col in df.columns if col != 'r2_score'], 
        color='r2_score',
        color_continuous_scale=px.colors.diverging.Spectral, # Using a more dynamic color scale
        color_continuous_midpoint=df['r2_score'].mean() # Center the color scale on the mean R2 score
    )
    fig.update_layout(title=title)
    fig.update_layout(
        title=title,
        title_x=0.25, # Başlığı yatayda ortalar
        title_y=0.99, # Başlığı grafiğin üstüne taşıyarak metin çakışmasını önler
        font_color="black",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=100, r=100, t=150, b=50)
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    # --- END OF MODIFICATION ---


def display_results(res):
    st.header("Analysis Results")
    plot_config_for_display = st.session_state.get('plot_config', DEFAULT_PLOT_STYLES)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 Performance Metrics", "Analysis Charts", "📊 Model Insights", "📉 Residuals & Location Insights", "🎯 Prediction Comparison", "✨ Pair Plots", "✨ Parallel Coordinates"]) 
    
    with tab1:
        st.info(f"**Model:** `{res['model_name']}` | **Split:** `{res['split_method']}` ({res['train_perc']}/{res['val_perc']}/{res['test_perc']}) | **Time Interval:** `{res['interval']}`")
        st.caption(f"Features Used: **{', '.join(res['features'])}**")
        
        st.subheader(f"Performance Metrics for {res['model_name']} Model (Training, Validation, Test Sets)")
        metrics_data = {
            "Metric": ["RMSE", "MAE", "MAPE", "R²"],
            "Training Set": [
                res['train_metrics']['rmse'], res['train_metrics']['mae'], 
                res['train_metrics']['mape'], res['train_metrics']['r2']
            ],
            "Validation Set": [
                res['val_metrics']['rmse'], res['val_metrics']['mae'], 
                res['val_metrics']['mape'], res['val_metrics']['r2']
            ],
            "Test Set": [
                res['test_metrics']['rmse'], res['test_metrics']['mae'], 
                res['test_metrics']['mape'], res['test_metrics']['r2']
            ],
        }
        df_metrics = pd.DataFrame(metrics_data).set_index("Metric")
        st.dataframe(df_metrics.style.format("{:.4f}"))

        st.markdown("---")
        st.subheader("Train/Validation/Test Set Distributions")
        plot_library_dist = st.radio("Select Plotting Library for Distributions", ["Plotly", "Seaborn"], key="dist_plot_library")
        if all(key in res for key in ['y_train', 'y_val', 'y_test']):
            if plot_library_dist == "Seaborn" and _SEABORN_AVAILABLE:
                plot_dataset_distributions_seaborn(
                    y_train=res['y_train'], 
                    y_val=res['y_val'], 
                    y_test=res['y_test'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_dataset_distributions(
                    y_train=res['y_train'], 
                    y_val=res['y_val'], 
                    y_test=res['y_test'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
        else:
            st.info("Dataset distribution data is not available.")
        st.markdown("---")

        st.subheader("Overall Test Performance (Summary)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall R² Score", f"{r2_score(res['y_test'], res['y_pred_series']):.4f}")
        col2.metric("Overall RMSE", f"{np.sqrt(mean_squared_error(res['y_test'], res['y_pred_series'])):.2f}")
        col3.metric("Overall MAE", f"{mean_absolute_error(res['y_test'], res['y_pred_series']):.2f}")
        
        st.subheader("Location-Based Test Performance")
        if res.get('detailed_loc_metrics_df') is not None and not res['detailed_loc_metrics_df'].empty:
            # Pivot the detailed table for better display
            display_df = res['detailed_loc_metrics_df'].pivot_table(
                index='Location', 
                columns='Set', 
                values=['r2', 'rmse', 'mae', 'mape']
            ).sort_index(axis=1, level=1) # Sort columns for consistent order
            # Improve column names for readability
            display_df.columns = [f'{val[1].capitalize()} {val[0].upper()}' for val in display_df.columns]
            st.dataframe(display_df.style.format("{:.2f}"))
        else:
            st.info("No location-based test results available.")


        st.subheader("Optimized Hyperparameters")
        if res.get('optimized', False) and res.get('best_params'):
            # Show optimization mode and parameters
            opt_mode_str = res.get('optimization_mode', 'N/A')
            st.write(f"Optimization Mode: **{opt_mode_str}**")
            
            df_params = pd.DataFrame([res['best_params']]).T.reset_index()
            df_params.columns = ["Parameter", "Value"]
            df_params['Value'] = df_params['Value'].astype(str) # PyArrow Error Fix
            st.dataframe(df_params)
        else:
            st.info("Model optimization was not enabled for this run, or no parameters were selected for optimization.")
        
        st.subheader("Default (Unoptimized) Parameters")
        if res.get('model'):
            current_model_params = res['model'].get_params()
            optimized_params_keys = res['best_params'].keys() if res.get('optimized', False) and res.get('best_params') else set()

            default_params_to_display = {}
            for param_name, param_value in current_model_params.items():
                if '__' in param_name or param_name == 'estimator':
                    continue
                
                if param_name not in optimized_params_keys:
                    if not isinstance(param_value, (pd.DataFrame, pd.Series, np.ndarray, list, dict)) or (isinstance(param_value, (list, dict)) and len(str(param_value)) < 50):
                        default_params_to_display[param_name] = param_value
                    else:
                        default_params_to_display[param_name] = f"<{type(param_value).__name__} object>"


            if default_params_to_display:
                df_default_params = pd.DataFrame([default_params_to_display]).T.reset_index()
                df_default_params.columns = ["Parameter", "Value"]
                df_default_params['Value'] = df_default_params['Value'].astype(str) # PyArrow Error Fix
                st.dataframe(df_default_params)
            else:
                if not res.get('optimized', False) or not res.get('best_params'):
                    st.info("All parameters are at their default values (optimization not enabled or no parameters selected for optimization).")
                else:
                    st.info("All relevant parameters were optimized, or default parameters are implicitly used.")
        else:
            st.info("Model object not available to retrieve default parameters.")

        st.subheader("Computational Load Summary")
        num_combinations_tried = "N/A"
        if res.get('optimized', False) and res.get('cv_results_'):
            try:
                num_combinations_tried = len(res['cv_results_']['params'])
            except KeyError:
                num_combinations_tried = "Error counting"
        
        total_analysis_duration_per_sample = "N/A"
        if res.get('analysis_duration') is not None and res.get('df_processed', pd.DataFrame()).shape[0] > 0:
            total_analysis_duration_per_sample = (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000

        avg_fit_time_per_optimized_combination = "N/A"
        if res.get('optimized', False) and res.get('cv_results_') and num_combinations_tried != "N/A" and num_combinations_tried > 0:
            avg_fit_time_per_optimized_combination = np.mean(res['cv_results_']['mean_fit_time'])
            
        comp_load_data = {
            "Metric/Parameter": [
                "Total Analysis Duration",
                "Processed Data Rows",
                "Processed Data Columns",
                "Model Input Features (for prediction)",
                "Selected Side Effects (Features)",
                "Selected Model",
                "Optimization Status",
                "Hyperparameter Combinations Tried",
                "Avg Fit Time per Opt. Combination",
                "Total Analysis Duration per Sample",
                "Sampling Period",
                "Data Split Ratio",
                "Splitting Method"
            ],
            "Value": [
                (f"{st.session_state.get('analysis_duration'):.2f} seconds"
                 if st.session_state.get('analysis_duration') is not None else "N/A"),
                str(res.get('df_processed', pd.DataFrame()).shape[0]),
                str(res.get('df_processed', pd.DataFrame()).shape[1]),
                str(len(res.get('feature_names_for_model', []))),
                (lambda features_list: ", ".join(f for f in features_list if f in ["Temperature", "Humidity"]) 
                 if any(item in ["Temperature", "Humidity"] for item in features_list) else "None")(res.get('features', [])),
                res.get('model_name', 'N/A'),
                "Performed" if res['optimized'] else "Not Performed",
                str(num_combinations_tried),
                f"{avg_fit_time_per_optimized_combination:.4f} seconds" if isinstance(avg_fit_time_per_optimized_combination, float) else avg_fit_time_per_optimized_combination,
                f"{total_analysis_duration_per_sample:.4f} ms/sample" if isinstance(total_analysis_duration_per_sample, float) else total_analysis_duration_per_sample,
                res.get('interval', 'N/A'),
                f"{res.get('train_perc', 'N/A')}/{res.get('val_perc', 'N/A')}/{res.get('test_perc', 'N/A')}%",
                res.get('split_method', 'N/A')
            ]
        }
        df_comp_load = pd.DataFrame(comp_load_data).set_index("Metric/Parameter")
        st.dataframe(df_comp_load)


    with tab2:
        st.header("Graphical Analysis")
        plot_loc_options = ['Overall'] + sorted(list(res['df_processed']['location'].unique()))
        plot_loc = st.selectbox("Sensor Location", options=plot_loc_options, key="tab2_plot_loc_select")
        plot_library = st.radio("Select Plotting Library", ["Plotly", "Seaborn"], key="main_plot_library")
        
        if plot_loc != 'Overall':
            all_indices_for_plot_loc = res['df_processed'][res['df_processed']['location'] == plot_loc].index
            plot_indices = all_indices_for_plot_loc.intersection(res['y_test'].index)
        else:
            plot_indices = res['y_test'].index
        
        if not plot_indices.empty:
            y_test_plot = res['y_test'].loc[plot_indices]
            y_pred_plot = res['y_pred_series'].loc[plot_indices]
            
            st.subheader(plot_config_for_display["scatter_plot"]["title"] or f"{res['model_name']} Model Correlation Plot - {plot_loc.capitalize()} {res['pollutant_unit']}")
            if plot_library == "Seaborn" and _SEABORN_AVAILABLE:
                plot_scatter_seaborn(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display, 
                    chart_key="scatter_tab2_seaborn"
                )
            else:
                plot_scatter(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display, 
                    chart_key="scatter_tab2_plotly"
                )
            
            st.subheader(plot_config_for_display["residuals_plot"]["title"] or f"{res['model_name']} Model Residuals Plot - {plot_loc.capitalize()} {res['pollutant_unit']}")
            if plot_library == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_seaborn(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals(
                    y_test=y_test_plot, 
                    y_pred=y_pred_plot, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            
            st.subheader(plot_config_for_display["time_series"]["title"] or f"{res['model_name']} Model Time Series - {plot_loc.capitalize()} {res['pollutant_unit']}")
            plot_df = res['df_processed'].loc[plot_indices].copy()
            plot_df['calibrated_pollutant'] = y_pred_plot
            if len(plot_df) > 500:
                plot_df = plot_df.sample(n=500, random_state=42).sort_values(by='timestamp')
            plot_time_series(
                df_plot=plot_df, 
                pollutant_unit=res['pollutant_unit'], 
                location_name=plot_loc, 
                model_name=res['model_name'], 
                display_unit=res['display_unit'], 
                plot_config=plot_config_for_display
            )
        else:
            st.info(f"No test data available for location '{plot_loc}' to display charts.")

    
    with tab3:
        st.header("Model Insights")

        if res['model_name'] in ["Random Forest", "Gradient Boosting", "Decision Tree", "AdaBoost", "XGBoost", "LightGBM", "CatBoost"]:
            st.subheader("Feature Importance")
            if res.get('model') and hasattr(res['model'], 'feature_importances_') and res.get('feature_names_for_model'):
                feature_importances = pd.Series(res['model'].feature_importances_, index=res['feature_names_for_model'])
                fig_fi = px.bar(
                    feature_importances.sort_values(ascending=False),
                    x=feature_importances.sort_values(ascending=False).index,
                    y=feature_importances.sort_values(ascending=False).values,
                    labels={'x': 'Feature', 'y': 'Importance'},
                    title=f"Feature Importance for {res['model_name']}",
                    text_auto=True
                )
                fig_fi.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig_fi, use_container_width=True, config=PLOTLY_CONFIG)
            else:
                st.info(f"Feature importance is not available for {res['model_name']} without a trained model or if the model does not expose feature_importances_. Ensure a model was trained successfully.")
        else:
            st.info(f"Feature Importance is typically applicable to tree-based models like Random Forest, Gradient Boosting, Decision Tree, AdaBoost, XGBoost, LightGBM, and CatBoost. Selected model: {res['model_name']}.")

        st.markdown("---")

        st.subheader("Hyperparameter Optimization Heatmap (if applicable)")
        if res.get('optimized', False) and res.get('cv_results_') and res.get('optimization_mode') not in ["Bayesian Optimization (skopt)", "RandomizedSearchCV"]:
            cv_results_df = pd.DataFrame(res['cv_results_'])
            optimized_param_keys = [p.replace('param_', '') for p in cv_results_df.columns if p.startswith('param_')]
            
            if len(optimized_param_keys) == 2:
                param1 = optimized_param_keys[0]
                param2 = optimized_param_keys[1]
                heatmap_data = cv_results_df.pivot_table(
                    values='mean_test_score', 
                    index=f'param_{param1}', 
                    columns=f'param_{param2}'
                )
                fig_heatmap = go.Figure(data=go.Heatmap(
                       z=heatmap_data.values,
                       x=heatmap_data.columns.astype(str),
                       y=heatmap_data.index.astype(str),
                       colorscale='Viridis',
                       colorbar=dict(title='Mean R² Score')
                   ))
                fig_heatmap.update_layout(
                    title=f'Hyperparameter Optimization Heatmap (R² Score) for {res["model_name"]}',
                    xaxis_title=param2,
                    yaxis_title=param1,
                    xaxis_nticks=len(heatmap_data.columns),
                    yaxis_nticks=len(heatmap_data.index)
                )
                st.plotly_chart(fig_heatmap, use_container_width=True, config=PLOTLY_CONFIG)
            elif len(optimized_param_keys) > 2:
                st.warning(f"Heatmap visualization is best for 2 optimized hyperparameters. {len(optimized_param_keys)} parameters were optimized. Please refer to performance metrics for details.")
            elif len(optimized_param_keys) < 2:
                 st.info("No heatmap generated: Less than 2 hyperparameters were selected for optimization.")
        elif res.get('optimization_mode') == "Bayesian Optimization (skopt)":
            st.info("Heatmap visualization is not applicable for Bayesian Optimization, as it does not search on a regular grid.")
        elif res.get('optimization_mode') == "RandomizedSearchCV":
             st.info("Heatmap visualization is not applicable for RandomizedSearchCV, as it does not search on a regular grid.")
        else:
            st.info("Hyperparameter optimization heatmap is available when GridSearch is enabled and at least 2 parameters are optimized.")


    with tab4:
        st.header("Residuals and Location-Based Insights")
        plot_loc_options_res = ['Overall'] + sorted(list(res['df_processed']['location'].unique()))
        plot_loc_res = st.selectbox("Sensor Location for Residuals Plots", options=plot_loc_options_res, key="tab4_plot_loc_select")
        plot_library_res = st.radio("Select Plotting Library for Residuals", ["Plotly", "Seaborn"], key="res_plot_library")

        if plot_loc_res != 'Overall':
            all_indices_for_plot_loc_res = res['df_processed'][res['df_processed']['location'] == plot_loc_res].index
            plot_indices_res = all_indices_for_plot_loc_res.intersection(res['y_test'].index)
        else:
            plot_indices_res = res['y_test'].index
        
        if not plot_indices_res.empty:
            y_test_plot_res = res['y_test'].loc[plot_indices_res]
            y_pred_plot_res = res['y_pred_series'].loc[plot_indices_res]

            st.subheader("Residuals Distribution (Histogram)")
            if plot_library_res == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_histogram_seaborn(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals_histogram(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )

            st.subheader("Residuals Density (KDE Plot)")
            if plot_library_res == "Seaborn" and _SEABORN_AVAILABLE:
                plot_residuals_kde_seaborn(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
            else:
                plot_residuals_kde(
                    y_test=y_test_plot_res, 
                    y_pred=y_pred_plot_res, 
                    pollutant_unit=res['pollutant_unit'], 
                    location_name=plot_loc_res, 
                    model_name=res['model_name'], 
                    display_unit=res['display_unit'], 
                    plot_config=plot_config_for_display
                )
        else:
            st.info(f"No test data available for location '{plot_loc_res}' to display residuals plots.")
        
        st.markdown("---")
        st.subheader("Location-Based Metric Consistency Across Datasets")
        if res.get('detailed_loc_metrics_df') is not None and not res['detailed_loc_metrics_df'].empty:
            detailed_df = res['detailed_loc_metrics_df']
            
            metric_options = {'RMSE': 'rmse', 'MAE': 'mae', 'R²': 'r2', 'MAPE': 'mape'}
            selected_metric_label = st.selectbox(
                "Select Metric to Compare",
                options=list(metric_options.keys()),
                key="location_consistency_metric_select"
            )
            selected_metric_col = metric_options[selected_metric_label]

            fig = px.box(
                detailed_df,
                x='Location',
                y=selected_metric_col,
                color='Location',
                title=f'Consistency of {selected_metric_label} Across Datasets for Each Location',
                labels={'Location': 'Location', selected_metric_col: selected_metric_label},
                points='all' # Show all 3 points
            )
            fig.update_layout(
                title=dict( # Yeni eklenen başlık ayarı
                    text=f'Consistency of {selected_metric_label} Across Datasets for Each Location',
                    font=dict(
                        size=plot_config_for_display["general"]["plot_title_font_size"],
                        family=plot_config_for_display["general"]["font_family"],
                        color=plot_config_for_display["general"]["font_color"]
                    )
                ),
                template=plot_config_for_display["general"]["template"],
                font=dict(family=plot_config_for_display["general"]["font_family"], color=plot_config_for_display["general"]["font_color"]),
                showlegend=False,
                xaxis=dict( # Eksen ayarları
                    title=dict( # Eksen başlığı font ayarı
                        text='Location',
                        font=dict(
                            size=plot_config_for_display["general"]["axis_title_font_size"],
                            color=plot_config_for_display["general"]["axis_title_font_color"],
                            family=plot_config_for_display["general"]["axis_title_font_family"]
                        )
                    ),
                    tickfont=dict( # Eksen tik etiketleri font ayarı
                        size=plot_config_for_display["general"]["axis_tick_font_size"],
                        color=plot_config_for_display["general"]["axis_tick_font_color"],
                        family=plot_config_for_display["general"]["axis_tick_font_family"]
                    )
                ),
                yaxis=dict( # Eksen ayarları
                    title=dict( # Eksen başlığı font ayarı
                        text=selected_metric_label,
                        font=dict(
                            size=plot_config_for_display["general"]["axis_title_font_size"],
                            color=plot_config_for_display["general"]["axis_title_font_color"],
                            family=plot_config_for_display["general"]["axis_title_font_family"]
                        )
                    ),
                    tickfont=dict( # Eksen tik etiketleri font ayarı
                        size=plot_config_for_display["general"]["axis_tick_font_size"],
                        color=plot_config_for_display["general"]["axis_tick_font_color"],
                        family=plot_config_for_display["general"]["axis_tick_font_family"]
                    )
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, key=f"loc_consistency_chart_{selected_metric_label}")
        else:
            st.info("No detailed location-based metrics available to plot.")

    with tab5:
        st.header("Model Predictions vs. Reference Values Across Data Splits")
        if all(key in res for key in ['y_train', 'y_train_pred_series', 'y_val', 'y_val_pred_series', 
                                      'y_test', 'y_pred_series', 'X_train', 'X_val', 'X_test', 'df_processed']):

            plot_loc_options_tab5 = ['Overall'] + sorted(list(res['df_processed']['location'].unique()))
            plot_loc_tab5 = st.selectbox(
                "Sensor Location", 
                options=plot_loc_options_tab5, 
                key="tab5_plot_loc_select"
            )

            if plot_loc_tab5 == 'Overall':
                y_train_plot = res['y_train']
                y_train_pred_plot = res['y_train_pred_series']
                y_val_plot = res['y_val']
                y_val_pred_plot = res['y_val_pred_series']
                y_test_plot = res['y_test']
                y_test_pred_plot = res['y_pred_series']
            else:
                train_indices = res['X_train'][res['X_train']['location'] == plot_loc_tab5].index
                y_train_plot = res['y_train'].loc[train_indices]
                y_train_pred_plot = res['y_train_pred_series'].loc[train_indices]

                val_indices = res['X_val'][res['X_val']['location'] == plot_loc_tab5].index
                y_val_plot = res['y_val'].loc[val_indices]
                y_val_pred_plot = res['y_val_pred_series'].loc[val_indices]

                test_indices = res['X_test'][res['X_test']['location'] == plot_loc_tab5].index
                y_test_plot = res['y_test'].loc[test_indices]
                y_test_pred_plot = res['y_pred_series'].loc[test_indices]
            
            plot_library_pred_comp = st.radio("Select Plotting Library for Prediction Comparison", ["Plotly", "Seaborn"], key="pred_comp_plot_library")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Training Set")
                if not y_train_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_train_plot, y_pred=y_train_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_train_plot, 
                            y_pred=y_train_pred_plot,
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_train"
                        )
                else:
                    st.info(f"No training data available for location: {plot_loc_tab5}")

            with col2:
                st.subheader("Validation Set")
                if not y_val_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_val_plot, y_pred=y_val_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_val_plot, 
                            y_pred=y_val_pred_plot, 
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_val"
                        )
                else:
                    st.info(f"No validation data available for location: {plot_loc_tab5}")

            with col3:
                st.subheader("Test Set")
                if not y_test_plot.empty:
                    if plot_library_pred_comp == "Seaborn" and _SEABORN_AVAILABLE:
                         plot_scatter_seaborn(y_test=y_test_plot, y_pred=y_test_pred_plot, pollutant_unit=res['pollutant_unit'], location_name=plot_loc_tab5, model_name=res['model_name'], display_unit=res['display_unit'], plot_config=plot_config_for_display)
                    else:
                        plot_scatter(
                            y_test=y_test_plot, 
                            y_pred=y_test_pred_plot,
                            pollutant_unit=res['pollutant_unit'], 
                            location_name=plot_loc_tab5,
                            model_name=res['model_name'], 
                            display_unit=res['display_unit'],
                            plot_config=plot_config_for_display,
                            chart_key="scatter_tab5_test"
                        )
                else:
                    st.info(f"No test data available for location: {plot_loc_tab5}")
        else:
            st.info("Training, Validation, or Test prediction data is not available to display comparison charts. Please run an analysis first.")

    with tab6:
        st.header("Hyperparameter Pair Plots")
        if res.get('optimized', False) and res.get('cv_results_'):
            cv_results_df = pd.DataFrame(res['cv_results_'])
            cv_results_df['r2_score'] = cv_results_df['mean_test_score']
            param_cols = [col for col in cv_results_df.columns if col.startswith('param_')]
            
            if len(param_cols) > 1:
                plot_df = cv_results_df[param_cols + ['r2_score']].rename(columns=lambda x: x.replace('param_', ''))
                
                if res.get('param_scope') == 'Top 3 Parameters' and res.get('optimization_mode') != "Manual Optimization":
                    top_3_keys = AUTO_OPTIMIZE_PARAMS.get(res['model_name'], [])
                    valid_top_3_params = [p for p in top_3_keys if p in plot_df.columns]
                    plot_df = plot_df[valid_top_3_params + ['r2_score']]
                
                if len(plot_df.columns) <= 6: # Heuristic limit for readability
                    if _SEABORN_AVAILABLE:
                        st.info("Using Seaborn for Pair Plot visualization for a different style.")
                        plot_pair_plots_seaborn(
                            plot_df,
                            title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                            plot_config=plot_config_for_display
                        )
                    else:
                        st.warning("Seaborn library not found. Falling back to default Plotly Pair Plots.")
                        plot_pair_plots(
                            plot_df,
                            title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                            plot_config=plot_config_for_display
                        )
                else:
                    st.warning(f"Pair plots can become difficult to read with more than 5 parameters. Your analysis has {len(plot_df.columns) - 1} parameters.")
                    
                    st.info("To improve readability, consider filtering parameters for the Pair Plot.")
                    all_params = [p for p in plot_df.columns if p != 'r2_score']
                    selected_params_pair = st.multiselect(
                        "Select parameters to display in Pair Plot:",
                        options=all_params,
                        default=all_params[:5] if len(all_params) > 5 else all_params,
                        key="pair_plot_params_multiselect"
                    )
                    
                    if selected_params_pair:
                        filtered_plot_df = plot_df[selected_params_pair + ['r2_score']]
                        if _SEABORN_AVAILABLE:
                            plot_pair_plots_seaborn(
                                filtered_plot_df,
                                title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                                plot_config=plot_config_for_display
                            )
                        else:
                            plot_pair_plots(
                                filtered_plot_df,
                                title=f"Hyperparameter Pair Plots for {res['model_name']} (R² Score)",
                                plot_config=plot_config_for_display
                            )

            else:
                st.info("Pair plots require at least 2 optimized hyperparameters.")
        else:
            st.info("No optimization results available to plot. Please run an analysis with optimization enabled.")

    with tab7:
        st.header("Hyperparameter Parallel Coordinates")
        if res.get('optimized', False) and res.get('cv_results_'):
            cv_results_df = pd.DataFrame(res['cv_results_'])
            cv_results_df['r2_score'] = cv_results_df['mean_test_score']
            param_cols = [col for col in cv_results_df.columns if col.startswith('param_')]
            
            if len(param_cols) > 0:
                plot_df = cv_results_df.rename(columns=lambda x: x.replace('param_', ''))
                plot_df['r2_score'] = plot_df['mean_test_score']

                valid_params = [col for col in plot_df.columns if plot_df[col].dtype in ['int64', 'float64', 'object'] and col != 'r2_score']
                
                st.info("Please select the parameters you want to view in the Parallel Coordinates Plot for better readability.")
                selected_params_parallel = st.multiselect(
                    "Select Parameters to Display:",
                    options=valid_params,
                    default=valid_params,
                    key="parallel_plot_params_multiselect"
                )
                
                if selected_params_parallel:
                    selected_plot_df = plot_df[selected_params_parallel + ['r2_score']]
                    plot_parallel_coordinates(
                        selected_plot_df,
                        title=f"Hyperparameter Parallel Coordinates for {res['model_name']} (R² Score)",
                        plot_config=plot_config_for_display
                    )
                else:
                    st.warning("Please select at least one parameter to display the plot.")
            else:
                st.info("Parallel Coordinates plot requires at least 1 optimized hyperparameter.")
        else:
            st.info("No optimization results available to plot. Please run an anaylsis with optimization enabled.")


def run_comparative_optimization_analysis(model_name, optimizers, config, pollutant_data, temp_data, hum_data, progress_bar_obj, status_text_container):
    results = []
    cv_results_dict = {}
    total_optimizers = len(optimizers)
    
    # Run data preparation once for all analyses
    status_text_container.info(f"[{model_name}] Data preparation for comparative analysis...")
    df_long_format, pollutant_unit, display_unit = merge_and_prepare_data(pollutant_data, temp_data, hum_data)
    
    resample_code = {'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min', '60 Minutes': '60min'}.get(config['interval_label'])
    if resample_code:
        df_processed = df_long_format.set_index('timestamp').groupby('location').resample(resample_code).mean().dropna().reset_index()
    else:
        df_processed = df_long_format

    if df_processed.empty:
        status_text_container.warning(f"[{model_name}] No data found for the selected interval.")
        return pd.DataFrame(), {}, None, None

    X_full = df_processed; y_full = df_processed['reference_pollutant']
    test_perc = 100 - config['train_perc'] - config['val_perc']

    # Split data only once
    if config['split_method'] == "Time-Based":
        df_sorted = X_full.sort_values(by='timestamp', ascending=True)
        train_end = int(len(df_sorted) * (config['train_perc'] / 100)); val_end = train_end + int(len(df_sorted) * (config['val_perc'] / 100))
        train_df, val_df, test_df = df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
    else:
        groups = df_processed['location']
        gss_test = GroupShuffleSplit(n_splits=1, test_size=(test_perc/100), random_state=42)
        train_val_idx, test_idx = next(gss_test.split(df_processed, groups=groups))
        train_val_df = df_processed.iloc[train_val_idx]
        test_df = df_processed.iloc[test_idx]
        val_size_corrected = config['val_perc'] / (config['train_perc'] + config['val_perc'])
        groups_val = train_val_df['location']
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_corrected, random_state=42)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=groups_val))
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    y_train = train_df['reference_pollutant']
    y_val = val_df['reference_pollutant']
    y_test = test_df['reference_pollutant']
    
    features_to_use_in_model = ['raw_pollutant']
    if config['use_temp']: features_to_use_in_model.append('raw_temp')
    if config['use_hum']: features_to_use_in_model.append('raw_humidity')

    X_train_dummies = pd.get_dummies(train_df[features_to_use_in_model])
    X_val_dummies = pd.get_dummies(val_df[features_to_use_in_model])
    X_test_dummies = pd.get_dummies(test_df[features_to_use_in_model])

    common_cols = list(set(X_train_dummies.columns) | set(X_val_dummies.columns) | set(X_test_dummies.columns))
    X_train_dummies = X_train_dummies.reindex(columns=common_cols, fill_value=0)
    X_val_dummies = X_val_dummies.reindex(columns=common_cols, fill_value=0)
    X_test_dummies = X_test_dummies.reindex(columns=common_cols, fill_value=0)
    model_feature_names = X_train_dummies.columns.tolist()
    
    scaler = None
    if model_name in ["Support Vector Regression (SVR)", "k-Nearest Neighbors (kNN)", "SGD", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "MLP"]:
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_train_dummies)
        X_val_processed = scaler.transform(X_val_dummies)
        X_test_processed = scaler.transform(X_test_dummies)
    else:
        X_train_processed, X_val_processed, X_test_processed = X_train_dummies, X_val_dummies, X_test_dummies

    y_train_val = np.concatenate((y_train, y_val))
    X_train_val_processed = np.concatenate((X_train_processed, X_val_processed)) if scaler else pd.concat([X_train_dummies, X_val_dummies])
    split_index = [-1] * len(X_train_processed) + [0] * len(X_val_processed)
    pds = PredefinedSplit(test_fold=split_index)
    
    
    for i, opt_mode in enumerate(optimizers):
        if st.session_state.stop_auto_run:
            status_text_container.warning("Comparative analysis stopped by user.")
            break
            
        status_text_container.info(f"[{model_name}] Running {opt_mode} optimization...")
        progress_bar_obj.progress((i) / total_optimizers, text=f"Running {opt_mode} for {model_name}...")
        
        start_time_opt = time.time()
        
        model = MODELS[model_name]
        best_params = None
        cv_results = None

        try:
            param_space_for_run = {}
            if opt_mode == "GridSearchCV":
                param_space_for_run = COMMON_PARAM_GRID.get(model_name, {})
                search = GridSearchCV(estimator=model, param_grid=param_space_for_run, cv=pds, n_jobs=-1, scoring='r2', return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_
            
            elif opt_mode == "RandomizedSearchCV":
                if not _SCIPY_AVAILABLE:
                    st.warning(f"Skipping RandomizedSearchCV: `scipy` library is not available.")
                    continue
                param_space_for_run = RANDOM_PARAM_SPACES.get(model_name, {})
                search = RandomizedSearchCV(estimator=model, param_distributions=param_space_for_run, n_iter=config.get('n_iter_random', 10), cv=pds, n_jobs=-1, scoring='r2', random_state=42, return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_

            elif opt_mode == "Bayesian Optimization":
                if not _SKOPT_AVAILABLE:
                    st.warning(f"Skipping Bayesian Optimization: `skopt` library is not available.")
                    continue
                param_space_for_run = BAYES_PARAM_SPACES.get(model_name, {})
                search = BayesSearchCV(estimator=model, search_spaces=param_space_for_run, n_iter=config.get('n_iter_bayes', 10), cv=pds, n_jobs=-1, scoring='r2', random_state=42, return_train_score=True)
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_
                best_params = search.best_params_
                cv_results = search.cv_results_
            
            else: # No optimization
                model.fit(X_train_processed, y_train)
                cv_results = {'params': [{}], 'mean_test_score': [calculate_all_metrics(y_val, model.predict(X_val_processed))['r2']], 'mean_fit_time': [0], 'mean_train_score': [calculate_all_metrics(y_train, model.predict(X_train_processed))['r2']]}
                best_params = {}

            duration_opt = time.time() - start_time_opt
            
            # Use the best model from the search to predict on all sets
            y_train_pred = model.predict(X_train_processed)
            y_val_pred = model.predict(X_val_processed)
            y_test_pred = model.predict(X_test_processed)

            # Calculate full metrics for all sets
            train_metrics = calculate_all_metrics(y_train, y_train_pred)
            val_metrics = calculate_all_metrics(y_val, y_val_pred)
            test_metrics = calculate_all_metrics(y_test, y_test_pred)
            
            results.append({
                "Model_Name": model_name,
                "Optimization_Method": opt_mode,
                "Duration (s)": duration_opt,
                "Combinations_Tried": len(cv_results['params']),
                "Best_Params": str(best_params),
                "Train_R2": train_metrics['r2'],
                "Train_RMSE": train_metrics['rmse'],
                "Train_MAE": train_metrics['mae'],
                "Train_MAPE": train_metrics['mape'],
                "Val_R2": val_metrics['r2'],
                "Val_RMSE": val_metrics['rmse'],
                "Val_MAE": val_metrics['mae'],
                "Val_MAPE": val_metrics['mape'],
                "Test_R2": test_metrics['r2'],
                "Test_RMSE": test_metrics['rmse'],
                "Test_MAE": test_metrics['mae'],
                "Test_MAPE": test_metrics['mape']
            })
            
            # Store cv_results for plotting
            cv_results_dict[opt_mode] = pd.DataFrame(cv_results)

        except Exception as e:
            status_text_container.error(f"Error during {opt_mode} optimization for {model_name}: {e}")
            cv_results_dict[opt_mode] = None # Store None if it failed
    
    return pd.DataFrame(results), cv_results_dict, pollutant_unit, display_unit


def plot_optimization_progress(cv_results_dict, model_name):
    """
    Plots R² scores vs. iterations for each optimization method, marking the best score with a star.
    """
    st.subheader(f"R² Score Change During Optimization Process for {model_name}")

    for opt_method, cv_df in cv_results_dict.items():
        if cv_df is not None and 'mean_test_score' in cv_df.columns:
            fig = go.Figure()
            
            # Find the best score and its index
            best_score = cv_df['mean_test_score'].max()
            best_score_index = cv_df['mean_test_score'].idxmax()

            # Plot the main line
            fig.add_trace(go.Scatter(
                x=np.arange(len(cv_df)),
                y=cv_df['mean_test_score'],
                mode='lines+markers',
                name=f'{model_name} ({opt_method})',
                line=dict(color='green', width=2),
                marker=dict(color='green', size=6)
            ))
            
            # Add the red star marker for the best score
            fig.add_trace(go.Scatter(
                x=[best_score_index],
                y=[best_score],
                mode='markers',
                marker=dict(symbol='star', size=15, color='red', line=dict(width=1, color='red')),
                name=f'Best R² Score ({best_score:.4f})'
            ))

            fig.update_layout(
                title=f"Optimization Progress for {model_name} ({opt_method}) - R² Score",
                xaxis_title="Combination Number",
                yaxis_title="Validation R² Score",
                template="plotly_white",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info(f"Iteration data not available for {opt_method}. It might not be applicable or the optimization did not run successfully.")


def display_optimization_comparison_results(results_df, cv_results_dict, model_name, display_unit):
    st.header(f"📈 Optimization Comparison for {model_name}")

    if results_df.empty:
        st.warning("No results to display for comparative analysis.")
        return

    tab1, tab2 = st.tabs(["📊 Performance Comparison", "📈 Optimization Process"])

    with tab1:
        st.subheader("R² Score Comparison")
        col1, col2 = st.columns(2)
        with col1:
            fig_r2 = px.bar(results_df, x='Optimization_Method', y='Test_R2', color='Optimization_Method',
                            title=f"Test R² Scores by Optimization Method for {model_name}",
                            labels={'Test_R2': 'Test R² Score', 'Optimization_Method': 'Method'},
                            color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_r2.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_r2, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            st.subheader("Total Analysis Duration Comparison")
            fig_time = px.bar(results_df, x='Optimization_Method', y='Duration (s)', color='Optimization_Method',
                              title=f"Total Analysis Duration for {model_name}",
                              labels={'Duration (s)': 'Duration (s)', 'Optimization_Method': 'Method'},
                              color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_time.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_time, use_container_width=True, config=PLOTLY_CONFIG)
        
        st.markdown("---")
        st.subheader("Detailed Performance Comparison Table")
        
        # Reorder columns for better readability
        detailed_columns = [
            'Optimization_Method', 'Duration (s)', 'Combinations_Tried', 'Test_R2',
            'Test_RMSE', 'Test_MAE', 'Test_MAPE', 'Val_R2', 'Val_RMSE', 'Val_MAE',
            'Val_MAPE', 'Train_R2', 'Train_RMSE', 'Train_MAE', 'Train_MAPE',
            'Best_Params'
        ]
        
        # Ensure only existing columns are used to prevent errors if a metric is missing
        columns_to_display = [col for col in detailed_columns if col in results_df.columns]
        display_df = results_df[columns_to_display]
        
        st.dataframe(
            display_df.style.format({
                "Duration (s)": "{:.2f}",
                "Test_R2": "{:.4f}", "Test_RMSE": "{:.2f}", "Test_MAE": "{:.2f}", "Test_MAPE": "{:.2f}%",
                "Val_R2": "{:.4f}", "Val_RMSE": "{:.2f}", "Val_MAE": "{:.2f}", "Val_MAPE": "{:.2f}%",
                "Train_R2": "{:.4f}", "Train_RMSE": "{:.2f}", "Train_MAE": "{:.2f}", "Train_MAPE": "{:.2f}%"
            })
        )

    with tab2:
        plot_optimization_progress(cv_results_dict, model_name)

    
# --- Main Model Run Function ---
def run_model_analysis(current_config, pollutant_data, temp_data, hum_data, progress_bar_obj, status_text_container):
    """
    Performs a single model analysis with the specified configuration.
    This function will be used for both single runs and the automatic batch loop.
    """
    st.session_state.start_time = time.time()
    
    try:
        status_text_container.info(f"[{current_config['model_name']}] Initializing...")
        update_progress_bar_with_eta(progress_bar_obj, 0, f"[{current_config['model_name']}] Initializing...")

        df_long_format, pollutant_unit, display_unit = merge_and_prepare_data(pollutant_data, temp_data, hum_data)
        status_text_container.info(f"[{current_config['model_name']}] Reading and merging data...")
        update_progress_bar_with_eta(progress_bar_obj, 10, f"[{current_config['model_name']}] Reading and merging data...")
        
        resample_code = {'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min','60 Minutes': '60min'}.get(current_config['interval_label'])
        if resample_code:
            status_text_container.info(f"[{current_config['model_name']}] Resampling data...")
            update_progress_bar_with_eta(progress_bar_obj, 20, f"[{current_config['model_name']}] Resampling data...")
            df_processed = df_long_format.set_index('timestamp').groupby('location').resample(resample_code).mean().dropna().reset_index()
        else:
            df_processed = df_long_format
        if df_processed.empty: 
            status_text_container.warning(f"[{current_config['model_name']}] No data found for the selected interval.")
            return None
        
        status_text_container.info(f"[{current_config['model_name']}] Splitting datasets...")
        update_progress_bar_with_eta(progress_bar_obj, 30, f"[{current_config['model_name']}] Splitting datasets...")
        X_full = df_processed; y_full = df_processed['reference_pollutant']
        test_perc = 100 - current_config['train_perc'] - current_config['val_perc']
        
        y_train, y_val, y_test = None, None, None 
        X_train, X_val, X_test = None, None, None 

        if current_config['split_method'] == "Time-Based":
            df_sorted = X_full.sort_values(by='timestamp', ascending=True)
            train_end = int(len(df_sorted) * (current_config['train_perc'] / 100)); val_end = train_end + int(len(df_sorted) * (current_config['val_perc'] / 100))
            train_df, val_df, test_df = df_sorted.iloc[:train_end], df_sorted.iloc[train_end:val_end], df_sorted.iloc[val_end:]
            
            y_train = train_df['reference_pollutant']
            y_val = val_df['reference_pollutant']
            y_test = test_df['reference_pollutant']
            
            X_train = train_df.drop(columns=['reference_pollutant'])
            X_val = val_df.drop(columns=['reference_pollutant'])
            X_test = test_df.drop(columns=['reference_pollutant'])

        else: # For 'Random' option, LOCATION-BASED SAFE method
            groups = df_processed['location']

            gss_test = GroupShuffleSplit(n_splits=1, test_size=(test_perc/100), random_state=42)
            train_val_idx, test_idx = next(gss_test.split(df_processed, groups=groups))

            train_val_df = df_processed.iloc[train_val_idx]
            test_df = df_processed.iloc[test_idx]

            val_size_corrected = current_config['val_perc'] / (current_config['train_perc'] + current_config['val_perc'])
            groups_val = train_val_df['location']
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_corrected, random_state=42)
            train_idx, val_idx = next(gss_val.split(train_val_df, groups=groups_val))
            
            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]

            y_train = train_df['reference_pollutant']
            y_val = val_df['reference_pollutant']
            y_test = test_df['reference_pollutant']

            X_train = train_df.drop(columns=['reference_pollutant'])
            X_val = val_df.drop(columns=['reference_pollutant'])
            X_test = test_df.drop(columns=['reference_pollutant'])
        
        status_text_container.info(f"[{current_config['model_name']}] Preparing features for the model...")
        update_progress_bar_with_eta(progress_bar_obj, 40, f"[{current_config['model_name']}] Preparing features for the model...")
        
        features_to_use_in_model = ['raw_pollutant']
        feature_info_for_display = ["Raw Pollutant"] 
        if current_config['use_temp']: 
            features_to_use_in_model.append('raw_temp')
            feature_info_for_display.append("Temperature")
        if current_config['use_hum']: 
            features_to_use_in_model.append('raw_humidity')
            feature_info_for_display.append("Humidity")

        X_train_dummies = pd.get_dummies(X_train[features_to_use_in_model])
        X_val_dummies = pd.get_dummies(X_val[features_to_use_in_model])
        X_test_dummies = pd.get_dummies(X_test[features_to_use_in_model])
        
        common_cols = list(set(X_train_dummies.columns) | set(X_val_dummies.columns) | set(X_test_dummies.columns))
        X_train_dummies = X_train_dummies.reindex(columns=common_cols, fill_value=0)
        X_val_dummies = X_val_dummies.reindex(columns=common_cols, fill_value=0)
        X_test_dummies = X_test_dummies.reindex(columns=common_cols, fill_value=0)
        
        model_feature_names = X_train_dummies.columns.tolist()

        scaler = None
        # Scaling is important for these models
        if current_config['model_name'] in ["Support Vector Regression (SVR)", "k-Nearest Neighbors (kNN)", "SGD", "Ridge Regression", "Lasso Regression", "ElasticNet Regression", "MLP"]:
            scaler = StandardScaler()
            X_train_processed = scaler.fit_transform(X_train_dummies)
            X_val_processed = scaler.transform(X_val_dummies)
            X_test_processed = scaler.transform(X_test_dummies)
        else:
            X_train_processed, X_val_processed, X_test_processed = X_train_dummies, X_val_dummies, X_test_dummies
        
        model = MODELS[current_config['model_name']]
        best_params = None 
        cv_results = None
        
        param_space_for_run = {}

        if current_config['optimize']:
            y_train_val = np.concatenate((y_train, y_val))
            X_train_val_processed = np.concatenate((X_train_processed, X_val_processed)) if scaler else pd.concat([X_train_dummies, X_val_dummies])
            
            split_index = [-1] * len(X_train_processed) + [0] * len(X_val_processed)
            pds = PredefinedSplit(test_fold=split_index)
            
            # Select the correct parameter space based on optimization mode
            if current_config['optimization_mode'] == "Bayesian Optimization (skopt)":
                if _SKOPT_AVAILABLE:
                    param_space_for_run = BAYES_PARAM_SPACES.get(current_config['model_name'], {})
                else:
                    status_text_container.warning(f"[{current_config['model_name']}] `skopt` library not found. Reverting to default parameters.")
                    current_config['optimize'] = False
            
            elif current_config['optimization_mode'] == "RandomizedSearchCV":
                if _SCIPY_AVAILABLE:
                    param_space_for_run = RANDOM_PARAM_SPACES.get(current_config['model_name'], {})
                else:
                    status_text_container.warning(f"[{current_config['model_name']}] `scipy` library not found. Reverting to default parameters.")
                    current_config['optimize'] = False

            elif current_config['optimization_mode'] == "Automatic Optimization (Top 3 Params)":
                top_3_keys = AUTO_OPTIMIZE_PARAMS.get(current_config['model_name'], [])
                full_params_grid = COMMON_PARAM_GRID.get(current_config['model_name'], {})
                for key in top_3_keys:
                    if key in full_params_grid:
                        param_space_for_run[key] = full_params_grid[key]
                # param_scope is implicitly 'Top 3' here
                current_config['param_scope'] = "Top 3 Parameters"

            else: # Manual Optimization (GridSearch)
                full_params_grid = COMMON_PARAM_GRID.get(current_config['model_name'], {})
                param_space_for_run = {key: full_params_grid[key] for key in current_config['manual_params'] if key in full_params_grid}

            # --- NEW FEATURE: DYNAMIC PARAMETER SCOPE ---
            # If "All Parameters" is selected, we do nothing as param_space_for_run is already populated.
            # If "Top 3 Parameters" is selected for a non-auto mode, we filter the list.
            if current_config.get('param_scope') == "Top 3 Parameters" and current_config['optimization_mode'] not in ["Automatic Optimization (Top 3 Params)"]:
                top_3_keys = AUTO_OPTIMIZE_PARAMS.get(current_config['model_name'], [])
                filtered_param_space = {}
                for key in top_3_keys:
                    if key in param_space_for_run:
                        filtered_param_space[key] = param_space_for_run[key]
                param_space_for_run = filtered_param_space


            if not param_space_for_run:
                 status_text_container.warning(f"[{current_config['model_name']}] No parameters selected for optimization. Skipping and training with default parameters.")
                 current_config['optimize'] = False
            
            if current_config['optimize']:
                if current_config['optimization_mode'] == "Bayesian Optimization (skopt)":
                    status_text_container.info(f"[{current_config['model_name']}] Running Bayesian Optimization (BayesSearchCV)...")
                    search = BayesSearchCV(
                        estimator=model,
                        search_spaces=param_space_for_run,
                        cv=pds,
                        n_iter=current_config.get('n_iter_bayes', 10),
                        n_jobs=-1,
                        scoring='r2',
                        random_state=42
                    )
                elif current_config['optimization_mode'] == "RandomizedSearchCV":
                    status_text_container.info(f"[{current_config['model_name']}] Running RandomizedSearchCV...")
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_space_for_run,
                        n_iter=current_config.get('n_iter_random', 10),
                        cv=pds,
                        n_jobs=-1,
                        scoring='r2',
                        random_state=42
                    )
                else: # GridSearch (Manual or Automatic)
                    status_text_container.info(f"[{current_config['model_name']}] Running GridSearchCV optimization...")
                    search = GridSearchCV(estimator=model, param_grid=param_space_for_run, cv=pds, n_jobs=-1, scoring='r2')
                
                update_progress_bar_with_eta(progress_bar_obj, 65, f"[{current_config['model_name']}] Optimization running...") 
                search.fit(X_train_val_processed, y_train_val)
                model = search.best_estimator_ 
                best_params = search.best_params_ 
                cv_results = search.cv_results_ 
            else:
                 model.fit(X_train_processed, y_train)


        status_text_container.info(f"[{current_config['model_name']}] Generating reports...")
        update_progress_bar_with_eta(progress_bar_obj, 90, f"[{current_config['model_name']}] Generating reports...")
        
        y_train_pred = model.predict(X_train_processed).flatten()
        y_val_pred = model.predict(X_val_processed).flatten()
        y_test_pred = model.predict(X_test_processed).flatten()
        
        end_time = time.time()
        duration = end_time - st.session_state.start_time
        st.session_state.analysis_duration = duration

        detailed_loc_results = []
        unique_locations = df_processed['location'].unique()

        for loc in sorted(unique_locations):
            train_indices_loc = X_train[X_train['location'] == loc].index
            if not train_indices_loc.empty:
                y_pred_train_loc = pd.Series(y_train_pred, index=y_train.index).loc[train_indices_loc]
                metrics = calculate_all_metrics(y_train.loc[train_indices_loc], y_pred_train_loc)
                metrics.update({'Location': loc, 'Set': 'Training'})
                detailed_loc_results.append(metrics)
            val_indices_loc = X_val[X_val['location'] == loc].index
            if not val_indices_loc.empty:
                y_pred_val_loc = pd.Series(y_val_pred, index=y_val.index).loc[val_indices_loc]
                metrics = calculate_all_metrics(y_val.loc[val_indices_loc], y_pred_val_loc)
                metrics.update({'Location': loc, 'Set': 'Validation'})
                detailed_loc_results.append(metrics)
            test_indices_loc = X_test[X_test['location'] == loc].index
            if not test_indices_loc.empty:
                y_pred_test_loc = pd.Series(y_test_pred, index=y_test.index).loc[test_indices_loc]
                metrics = calculate_all_metrics(y_test.loc[test_indices_loc], y_pred_test_loc)
                metrics.update({'Location': loc, 'Set': 'Test'})
                detailed_loc_results.append(metrics)
        
        detailed_loc_metrics_df = pd.DataFrame(detailed_loc_results)

        analysis_results = { 
            "y_test": y_test, "y_pred_series": pd.Series(y_test_pred, index=y_test.index), 
            "X_test": X_test, "df_processed": df_processed, "model_name": current_config['model_name'], 
            "interval": current_config['interval_label'], "split_method": current_config['split_method'], 
            "train_perc": current_config['train_perc'], "val_perc": current_config['val_perc'], "test_perc": test_perc, 
            "features": feature_info_for_display, "pollutant_unit": pollutant_unit, "display_unit": display_unit, 
            "train_metrics": calculate_all_metrics(y_train, y_train_pred),
            "val_metrics": calculate_all_metrics(y_val, y_val_pred),
            "test_metrics": calculate_all_metrics(y_test, y_test_pred),
            "optimized": current_config['optimize'], "best_params": best_params, "cv_results_": cv_results, 
            "model": model, "feature_names_for_model": model_feature_names, "analysis_duration": duration,
            "detailed_loc_metrics_df": detailed_loc_metrics_df,
            "y_train": y_train, "y_train_pred_series": pd.Series(y_train_pred, index=y_train.index),
            "y_val": y_val, "y_val_pred_series": pd.Series(y_val_pred, index=y_val.index),
            "X_train": X_train, "X_val": X_val,
            "optimization_mode": current_config['optimization_mode'],
            "param_scope": current_config.get('param_scope')
        }
        
        update_progress_bar_with_eta(progress_bar_obj, 100, f"[{current_config['model_name']}] Analysis Complete!")
        return analysis_results

    except Exception as e:
        status_text_container.error(f"[{current_config['model_name']}] ERROR: An error occurred during analysis: {e}")
        progress_bar_obj.empty()
        return None

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_duration' not in st.session_state:
    st.session_state.analysis_duration = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'plot_config' not in st.session_state:
    st.session_state.plot_config = DEFAULT_PLOT_STYLES
if 'history' not in st.session_state:
    st.session_state.history = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0
if 'run_analysis_triggered' not in st.session_state:
    st.session_state.run_analysis_triggered = False
if 'run_all_models_triggered' not in st.session_state:
    st.session_state.run_all_models_triggered = False
if 'run_config' not in st.session_state:
    st.session_state.run_config = {}
if 'auto_optimize_param_keys_cache' not in st.session_state:
    st.session_state.auto_optimize_param_keys_cache = {}
if 'stop_auto_run' not in st.session_state:
    st.session_state.stop_auto_run = False
if 'run_optimization_comparison_triggered' not in st.session_state:
    st.session_state.run_optimization_comparison_triggered = False
if 'comparative_results' not in st.session_state:
    st.session_state['comparative_results'] = None
if 'comparative_model_name' not in st.session_state:
    st.session_state['comparative_model_name'] = None
if 'comparative_cv_results' not in st.session_state:
    st.session_state['comparative_cv_results'] = {}


MODELS_TO_RUN_FIRST = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet Regression",
    "SGD",
    "Decision Tree",
    "k-Nearest Neighbors (kNN)",
    "LightGBM",
    "XGBoost",
    "CatBoost",
    "AdaBoost",
    "Gradient Boosting",
    "Random Forest"
]

MODELS_TO_RUN_LAST = [
    "MLP",
    "Support Vector Regression (SVR)"
]


config = {}

with st.sidebar:
    st.title("Control Panel")
    
    with st.expander("1. Data Management", expanded=True):
        uploaded_file_pollutant = st.file_uploader("1. Upload Main Pollutant Dataset (CO2, PM2.5 etc.)", type=['csv'], key=f"pollutant_{st.session_state.uploader_key}")
        uploaded_file_temp = st.file_uploader("2. Upload Temperature Dataset", type=['csv'], key=f"temp_{st.session_state.uploader_key}")
        uploaded_file_hum = st.file_uploader("3. Upload Humidity Dataset", type=['csv'], key=f"hum_{st.session_state.uploader_key}")

    with st.expander("2. Model Configuration", expanded=True):
        config['model_name'] = st.selectbox("Select Machine Learning Model", options=list(MODELS.keys()), key="model_select_sidebar")
        config['interval_label'] = st.selectbox("Sampling Period", options={'1 Minute (Original)': None, '2 Minutes': '2min','3 Minutes': '3min','5 Minutes': '5min', '10 Minutes': '10min', '15 Minutes': '15min', '30 Minutes': '30min','60 Minutes': '60min'}.keys(), key="interval_select_sidebar")
        st.subheader("Environmental Parameters")
        config['use_temp'] = st.checkbox("Temperature", value=True, key="use_temp_sidebar")
        config['use_hum'] = st.checkbox("Humidity", value=True, key="use_hum_sidebar")
        st.subheader("Data Splitting Strategy")
        config['split_method'] = st.radio("Splitting Method", ["Time-Based", "Random"], index=0, key="split_method_sidebar")
        config['train_perc'] = st.slider("Training Set Ratio (%)", 10, 90, 70, 5, key="train_perc_sidebar")
        config['val_perc'] = st.slider("Validation Set Ratio (%)", 5, (100 - config['train_perc']), 15, 5, key="val_perc_sidebar")

        test_perc_sidebar = 100 - config['train_perc'] - config['val_perc']
        st.metric(label="Test Set Ratio (%)", value=f"{test_perc_sidebar}%", delta_color="off")

        config['optimize'] = st.checkbox("Optimize Model", help="May slow down the process!", key="optimize_sidebar")
        
        if config['optimize']:
            optimization_options = ["Manual Optimization", "Automatic Optimization (Top 3 Params)"]
            if _SKOPT_AVAILABLE:
                optimization_options.append("Bayesian Optimization (skopt)")
            if _SCIPY_AVAILABLE:
                optimization_options.append("RandomizedSearchCV")

            config['optimization_mode'] = st.radio(
                "Select Optimization Mode",
                optimization_options,
                index=0,
                key="optimization_mode_select"
            )
            
            if config['optimization_mode'] in ["RandomizedSearchCV", "Bayesian Optimization (skopt)"]:
                if config['optimization_mode'] == "RandomizedSearchCV":
                    config['n_iter_random'] = st.slider("Number of search iterations (n_iter)", 10, 200, 32, step=1, key="n_iter_random")
                elif config['optimization_mode'] == "Bayesian Optimization (skopt)":
                    if config['model_name'] in BAYES_PARAM_SPACES:
                        config['n_iter_bayes'] = st.slider("Number of search iterations (n_iter)", 10, 100, 32, step=1, key="n_iter_bayes")

                config['param_scope'] = st.radio(
                    "Optimize Parameter Scope",
                    options=["Top 3 Parameters", "All Parameters"],
                    index=1,
                    key="param_scope_radio"
                )

            elif config['optimization_mode'] == "Automatic Optimization (Top 3 Params)":
                 config['param_scope'] = "Top 3 Parameters"
                 st.info("Automatic Optimization will always use the top 3 parameters.")
            else: # Manual Optimization
                config['param_scope'] = st.radio(
                    "Optimize Parameter Scope",
                    options=["Top 3 Parameters", "All Parameters"],
                    index=0,
                    key="param_scope_radio"
                )
            
        else:
            config['optimization_mode'] = "None"
            config['param_scope'] = "None"

        selected_optimization_params = [] 
        if config['optimize'] and config['optimization_mode'] == "Manual Optimization":
            st.subheader(f"Select Parameters for {config['model_name']} Manual Optimization")
            model_potential_params = COMMON_PARAM_GRID.get(config['model_name'], {})
            
            if model_potential_params:
                default_params = []
                if config.get('param_scope') == "Top 3 Parameters":
                    default_params = AUTO_OPTIMIZE_PARAMS.get(config['model_name'], [])[:3]
                else:
                    default_params = list(model_potential_params.keys())
                
                selected_optimization_params = st.multiselect(
                    "Choose hyperparameters to optimize:",
                    options=list(model_potential_params.keys()),
                    default=default_params,
                    key=f"manual_optimize_params_multiselect_{config['model_name']}"
                )
                config['manual_params'] = selected_optimization_params

                if not selected_optimization_params:
                    st.warning("No parameters selected for manual optimization. GridSearch will not run for this model.")
            else:
                st.warning(f"No predefined parameters found for {config['model_name']} to optimize manually.")


    run_button_placeholder = st.empty()
    run_all_models_button_placeholder = st.empty()
    run_opt_comparison_button_placeholder = st.empty()

st.markdown("---")

if 'history' in st.session_state and st.session_state.history:
    with st.expander("📊 Analysis History (Comparison Table)", expanded=True):
        history_df = pd.DataFrame(st.session_state.history).sort_values(by="Analysis Time", ascending=False)
        
        column_order = [
            "Analysis Time", "Model Name", 
            "Analysis Duration (s)", "Time per Sample (ms)",
            "Training R²","Validation R²","Test R²",
            "Training RMSE","Validation RMSE", "Test RMSE", 
            "Training MAE","Validation MAE","Test MAE", 
            "Training MAPE","Validation MAPE","Test MAPE", 
            "Environmental Factors","Train %", "Val %", "Test %",
            "Opt. Status", "Opt. Mode", "Optimized Parameters", "Opt. Param Values",
            "Splitting Method", "Sampling Period",
            "Processed Rows", "Processed Columns", "Model Input Features"
        ]
        
        existing_columns_in_order = [col for col in column_order if col in history_df.columns]
        history_df_display = history_df[existing_columns_in_order]

        formatter = {
            "Training RMSE": "{:.3f}", "Training MAE": "{:.3f}", "Training MAPE": "{:.2f}%", "Training R²": "{:.4f}",
            "Validation RMSE": "{:.3f}", "Validation MAE": "{:.3f}", "Validation MAPE": "{:.2f}%", "Validation R²": "{:.4f}",
            "Test RMSE": "{:.3f}", "Test MAE": "{:.3f}", "Test MAPE": "{:.2f}%", "Test R²": "{:.4f}",
            "Analysis Duration (s)": "{:.2f}",
            "Time per Sample (ms)": "{:.4f}"
        }
        
        st.dataframe(history_df_display.style.format(formatter), use_container_width=True)
        
        if st.button("Clear Analysis History", key="clear_history", type="secondary"):
            st.session_state.history = []
            st.rerun()

if st.button("Start New Analysis (Reset All)", type="secondary"):
    st.session_state.pop('analysis_results', None)
    st.session_state.pop('analysis_duration', None)
    st.session_state.pop('start_time', None)
    st.session_state.uploader_key += 1
    st.session_state.run_analysis_triggered = False
    st.session_state.run_all_models_triggered = False
    st.session_state.run_optimization_comparison_triggered = False
    st.session_state['comparative_results'] = None
    st.session_state['comparative_model_name'] = None
    st.session_state['comparative_cv_results'] = {}
    st.rerun()

if uploaded_file_pollutant and uploaded_file_temp and uploaded_file_hum:
    if run_button_placeholder.button("⚡ Start Single Model Analysis / Update Report", type="primary"):
        st.session_state.run_analysis_triggered = True
        st.session_state.run_all_models_triggered = False
        st.session_state.run_optimization_comparison_triggered = False
        st.session_state.run_config = config
        st.session_state['comparative_results'] = None
        st.session_state['comparative_cv_results'] = {}
        st.rerun()
    
    if run_all_models_button_placeholder.button("🚀 Run All ML Models Automatically (Batch Analysis)", type="secondary"):
        st.session_state.run_all_models_triggered = True
        st.session_state.run_analysis_triggered = False
        st.session_state.run_optimization_comparison_triggered = False
        st.session_state.stop_auto_run = False
        st.session_state.base_auto_run_config = config.copy()
        st.session_state.auto_optimize_param_keys_cache = AUTO_OPTIMIZE_PARAMS
        st.session_state['comparative_results'] = None
        st.session_state['comparative_cv_results'] = {}
        st.rerun()

    if run_opt_comparison_button_placeholder.button("📊 Run Optimization Comparison (Grid vs. Random vs. Bayesian)", type="secondary"):
        st.session_state.run_optimization_comparison_triggered = True
        st.session_state.run_analysis_triggered = False
        st.session_state.run_all_models_triggered = False
        st.session_state.stop_auto_run = False
        st.session_state.opt_comp_config = config.copy()
        st.session_state['comparative_results'] = None # Önceki sonuçları temizle
        st.session_state['comparative_cv_results'] = {}
        st.rerun()
else:
    st.info("Please upload all 3 datasets from the **Control Panel** on the left to start the analysis.")


main_content_area = st.container() 
progress_status_container = main_content_area.empty()
stop_button_container = main_content_area.empty()


if st.session_state.run_analysis_triggered:
    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

    status_text_placeholder.info("Single model analysis in progress...")
    
    analysis_results = run_model_analysis(
        st.session_state.run_config, 
        uploaded_file_pollutant, 
        uploaded_file_temp, 
        uploaded_file_hum,
        progress_bar_placeholder,
        status_text_placeholder
    )
    
    if analysis_results:
        st.session_state['analysis_results'] = analysis_results
        
        res = analysis_results
        env_factors = [f for f in res['features'] if f != "Raw Pollutant"]
        env_factors_str = ", ".join(env_factors) if env_factors else "None"
        
        opt_param_values_str = "N/A"
        if res.get('optimized', False) and res.get('best_params'):
            if isinstance(res['best_params'], dict):
                params_to_display = dict(res['best_params'])
            else:
                params_to_display = res['best_params']
            opt_param_values_str = ", ".join([f"{k}: {v}" for k, v in params_to_display.items()])

        
        run_summary_detailed = {
            "Analysis Time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Model Name": res['model_name'],
            "Opt. Status": "Yes" if res['optimized'] else "No",
            "Opt. Mode": res.get('optimization_mode', 'N/A'),
            "Optimized Parameters": ", ".join(res['best_params'].keys()) if res.get('optimized', False) and res.get('best_params') else "None",
            "Opt. Param Values": opt_param_values_str,
            "Training R²": res['train_metrics']['r2'],
            "Validation R²": res['val_metrics']['r2'],
            "Test R²": res['test_metrics']['r2'],
            "Training RMSE": res['train_metrics']['rmse'],
            "Validation RMSE": res['val_metrics']['rmse'],
            "Test RMSE": res['test_metrics']['rmse'],
            "Training MAE": res['train_metrics']['mae'],
            "Validation MAE": res['val_metrics']['mae'],
            "Test MAE": res['test_metrics']['mae'],
            "Training MAPE": res['train_metrics']['mape'],
            "Validation MAPE": res['val_metrics']['mape'],
            "Test MAPE": res['test_metrics']['mape'],
            "Environmental Factors": env_factors_str,
            "Train %": res['train_perc'],
            "Val %": res['val_perc'],
            "Test %": res['test_perc'],
            "Analysis Duration (s)": res['analysis_duration'],
            "Time per Sample (ms)": (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000 if res['df_processed'].shape[0] > 0 else np.nan,
            "Sampling Period": res['interval'],
            "Splitting Method": res['split_method'],
            "Processed Rows": res['df_processed'].shape[0],
            "Processed Columns": res['df_processed'].shape[1],
            "Model Input Features": ", ".join(res.get('feature_names_for_model', []))
        }
        st.session_state.history.append(run_summary_detailed)
        status_text_placeholder.success(f"Analysis for {st.session_state.run_config['model_name']} completed in {st.session_state.analysis_duration:.2f} seconds.")
    
    st.session_state.run_analysis_triggered = False
    st.rerun()


elif st.session_state.run_all_models_triggered:
    total_models = len(MODELS_TO_RUN_FIRST) + len(MODELS_TO_RUN_LAST)
    combined_model_order = MODELS_TO_RUN_FIRST + MODELS_TO_RUN_LAST

    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()

    if stop_button_container.button("🛑 Stop Batch Analysis", key="stop_auto_run_batch_button", type="secondary"):
        st.session_state.stop_auto_run = True
        stop_button_container.empty()
        status_text_placeholder.warning("Stop signal received. Finishing current model, then stopping batch analysis.")


    for i, model_name in enumerate(combined_model_order):
        if st.session_state.stop_auto_run:
            status_text_placeholder.info("Batch analysis interrupted by user.")
            break

        current_progress_percent = int(((i) / total_models) * 100)
        status_text_placeholder.info(f"Running model {i+1}/{total_models}: {model_name}...")
        
        progress_bar_obj = progress_bar_placeholder.progress(current_progress_percent, text=f"Starting analysis for {model_name}...")
        
        st.session_state.start_time = time.time()

        temp_config = st.session_state.base_auto_run_config.copy()
        temp_config['model_name'] = model_name
        
        # In batch mode, manual optimization settings are not applicable
        if temp_config.get('optimization_mode') == "Manual Optimization":
             temp_config['optimize'] = False

        analysis_results = run_model_analysis(
            temp_config, 
            uploaded_file_pollutant, 
            uploaded_file_temp, 
            uploaded_file_hum,
            progress_bar_obj,
            status_text_placeholder
        )
        
        if analysis_results and not st.session_state.stop_auto_run:
            st.session_state['analysis_results'] = analysis_results
            
            res = analysis_results
            
            env_factors = [f for f in res['features'] if f != "Raw Pollutant"]
            env_factors_str = ", ".join(env_factors) if env_factors else "None"

            opt_param_values_str = "N/A"
            if res.get('optimized', False) and res.get('best_params'):
                if isinstance(res['best_params'], dict):
                    params_to_display = dict(res['best_params'])
                else:
                    params_to_display = res['best_params']
                opt_param_values_str = ", ".join([f"{k}: {v}" for k, v in params_to_display.items()])

            run_summary_detailed = {
                "Analysis Time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Model Name": res['model_name'],
                "Opt. Status": "Yes" if res['optimized'] else "No",
                "Opt. Mode": res.get('optimization_mode', 'N/A'),
                "Optimized Parameters": ", ".join(res['best_params'].keys()) if res.get('optimized', False) and res.get('best_params') else "None",
                "Opt. Param Values": opt_param_values_str,
                "Training R²": res['train_metrics']['r2'],
                "Validation R²": res['val_metrics']['r2'],
                "Test R²": res['test_metrics']['r2'],
                "Training RMSE": res['train_metrics']['rmse'],
                "Validation RMSE": res['val_metrics']['rmse'],
                "Test RMSE": res['test_metrics']['rmse'],
                "Training MAE": res['train_metrics']['mae'],
                "Validation MAE": res['val_metrics']['mae'],
                "Test MAE": res['test_metrics']['mae'],
                "Training MAPE": res['train_metrics']['mape'],
                "Validation MAPE": res['val_metrics']['mape'],
                "Test MAPE": res['test_metrics']['mape'],
                "Environmental Factors": env_factors_str,
                "Train %": res['train_perc'],
                "Val %": res['val_perc'],
                "Test %": res['test_perc'],
                "Analysis Duration (s)": res['analysis_duration'],
                "Time per Sample (ms)": (res['analysis_duration'] / res['df_processed'].shape[0]) * 1000 if res['df_processed'].shape[0] > 0 else np.nan,
                "Sampling Period": res['interval'],
                "Splitting Method": res['split_method'],
                "Processed Rows": res['df_processed'].shape[0],
                "Processed Columns": res['df_processed'].shape[1],
                "Model Input Features": ", ".join(res.get('feature_names_for_model', []))
            }
            st.session_state.history.append(run_summary_detailed)

    progress_bar_placeholder.empty()
    status_text_placeholder.empty()

    if i == total_models - 1 and analysis_results:
        status_text_placeholder.success("Batch analysis completed successfully!")
    else:
        status_text_placeholder.info("Batch analysis completed or interrupted by user.")
    
elif st.session_state.run_optimization_comparison_triggered:
    with progress_status_container:
        status_text_placeholder = st.empty()
        progress_bar_placeholder = st.empty()
        stop_button_container.button("🛑 Stop Comparison", key="stop_opt_comp_button", type="secondary")
    
    model_to_compare = st.session_state.opt_comp_config['model_name']
    optimizers_to_compare = ["GridSearchCV", "RandomizedSearchCV"]
    if _SKOPT_AVAILABLE:
        optimizers_to_compare.append("Bayesian Optimization")
        
    results_df, cv_results_dict, pollutant_unit, display_unit = run_comparative_optimization_analysis(
        model_to_compare,
        optimizers_to_compare,
        st.session_state.opt_comp_config,
        uploaded_file_pollutant,
        uploaded_file_temp,
        uploaded_file_hum,
        progress_bar_placeholder,
        status_text_placeholder
    )
    
    if results_df is not None:
        st.session_state['comparative_results'] = results_df
        st.session_state['comparative_cv_results'] = cv_results_dict
        st.session_state['comparative_model_name'] = model_to_compare
        st.session_state['comparative_display_unit'] = display_unit
        
    st.session_state.run_optimization_comparison_triggered = False
    st.rerun()


with main_content_area:
    if 'analysis_results' in st.session_state and st.session_state.analysis_results and not st.session_state.run_analysis_triggered and not st.session_state.run_all_models_triggered and not st.session_state.run_optimization_comparison_triggered:
        display_results(st.session_state.analysis_results)
        if st.session_state.analysis_duration is not None:
            st.success(f"Last analysis completed in {st.session_state.analysis_duration:.2f} seconds.")
    
    if 'comparative_results' in st.session_state and st.session_state.comparative_results is not None:
        if not st.session_state.comparative_results.empty:
            display_optimization_comparison_results(
                st.session_state.comparative_results, 
                st.session_state.comparative_cv_results,
                st.session_state.comparative_model_name,
                st.session_state.comparative_display_unit
            )