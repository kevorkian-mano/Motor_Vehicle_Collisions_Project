# -*- coding: utf-8 -*-
"""NYC Crash Dashboard - Render Deployment"""

import ast
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.figure_factory as ff
import os

# =============================================================================
# DATA LOADING - OPTIMIZED FOR RENDER
# =============================================================================

def load_data():
    """
    Load dataset with optimizations for Render free tier
    """
    try:
        # Load from your Google Drive with sampling for free tier
        print("🔄 Loading data from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1t6vLcjuFNNON9XHYNPow8uJYwFmFP48G"
        df = pd.read_csv(url, dtype=str, nrows=50000)
        print(f"✅ Successfully loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔄 Creating sample dataset...")
        return create_sample_dataset()

def create_sample_dataset():
    """
    Create sample dataset if download fails
    """
    np.random.seed(42)
    n_samples = 10000
    
    boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    vehicle_types = ['Sedan', 'SUV/Station Wagon', 'Taxi', 'Bus', 'Bicycle', 'Motorcycle']
    factors = ['Driver Inattention/Distraction', 'Following Too Closely', 'Failure to Yield Right-of-Way']
    
    dates = pd.date_range('2020-01-01', '2023-12-31', periods=n_samples)
    
    df = pd.DataFrame({
        'CRASH_DATETIME': dates,
        'BOROUGH': np.random.choice(boroughs, n_samples),
        'LATITUDE': np.random.uniform(40.50, 40.90, n_samples),
        'LONGITUDE': np.random.uniform(-74.25, -73.70, n_samples),
        'NUMBER OF PERSONS INJURED': np.random.randint(0, 5, n_samples),
        'NUMBER OF PERSONS KILLED': np.random.randint(0, 2, n_samples),
        'ALL_VEHICLE_TYPES': [str([np.random.choice(vehicle_types) for _ in range(np.random.randint(1, 3))]) for _ in range(n_samples)],
        'ALL_CONTRIBUTING_FACTORS': [str([np.random.choice(factors) for _ in range(np.random.randint(1, 2))]) for _ in range(n_samples)],
        'ON STREET NAME': [f"Street {i}" for i in range(n_samples)],
        'PERSON_TYPE': np.random.choice(['Pedestrian', 'Driver', 'Passenger', 'Cyclist'], n_samples),
        'PERSON_INJURY': np.random.choice(['Injured', 'Killed', 'Uninjured'], n_samples),
        'PERSON_AGE': np.random.randint(18, 80, n_samples),
        'PERSON_SEX': np.random.choice(['M', 'F'], n_samples),
        'SAFETY_EQUIPMENT': np.random.choice(['Seat Belt', 'Helmet', 'None'], n_samples),
        'EMOTIONAL_STATUS': np.random.choice(['Normal', 'Upset', 'Calm'], n_samples),
        'EJECTION': np.random.choice(['Not Ejected', 'Ejected'], n_samples),
        'COMPLAINT': np.random.choice(['Pain', 'Injury', 'None'], n_samples),
        'POSITION_IN_VEHICLE_CLEAN': np.random.choice(['Driver', 'Front Passenger', 'Rear Passenger'], n_samples),
        'ZIP CODE': np.random.randint(10001, 11698, n_samples)
    })
    
    df['UNIQUE_ID'] = range(1, len(df) + 1)
    print("✅ Sample dataset created for demo")
    return df

# Load the data
print("🚀 Starting NYC Crash Dashboard...")
df = load_data()

# =============================================================================
# DATA PROCESSING (YOUR EXISTING CODE)
# =============================================================================

# Clean borough names to proper capitalization
borough_mapping = {
    'MANHATTAN': 'Manhattan',
    'BROOKLYN': 'Brooklyn',
    'QUEENS': 'Queens',
    'BRONX': 'Bronx',
    'STATEN ISLAND': 'Staten Island'
}

if "BOROUGH" in df.columns:
    df["BOROUGH"] = df["BOROUGH"].str.title().replace(borough_mapping)
    df["BOROUGH"] = df["BOROUGH"].fillna("Unknown")
else:
    df["BOROUGH"] = "Unknown"

# Normalize and cast useful columns
df["CRASH_DATETIME"] = pd.to_datetime(df["CRASH_DATETIME"], errors="coerce")
df["YEAR"] = df["CRASH_DATETIME"].dt.year
df["MONTH"] = df["CRASH_DATETIME"].dt.month
df["HOUR"] = df["CRASH_DATETIME"].dt.hour
df["DAY_OF_WEEK"] = df["CRASH_DATETIME"].dt.day_name()

# Cast numeric injury/killed counts to numeric (safe)
num_cols = [
     "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED",
     "NUMBER OF PEDESTRIANS INJURED", "NUMBER OF PEDESTRIANS KILLED",
     "NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED",
     "NUMBER OF MOTORIST INJURED", "NUMBER OF MOTORIST KILLED"
]
for c in num_cols:
     if c in df.columns:
          df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
     else:
          df[c] = 0

# Helpful aggregated numeric columns
df["TOTAL_INJURED"] = df[["NUMBER OF PERSONS INJURED",
                         "NUMBER OF PEDESTRIANS INJURED",
                         "NUMBER OF CYCLIST INJURED",
                         "NUMBER OF MOTORIST INJURED"]].sum(axis=1)
df["TOTAL_KILLED"] = df[["NUMBER OF PERSONS KILLED",
                         "NUMBER OF PEDESTRIANS KILLED",
                         "NUMBER OF CYCLIST KILLED",
                         "NUMBER OF MOTORIST KILLED"]].sum(axis=1)

# Create severity score for advanced analysis
df["SEVERITY_SCORE"] = (df["TOTAL_INJURED"] * 1 + df["TOTAL_KILLED"] * 5)

# FULL_ADDRESS fallback
if "FULL ADDRESS" not in df.columns:
     df["FULL ADDRESS"] = df.get("ON STREET NAME", "").fillna("") + ", " + df.get("BOROUGH", "")

# Latitude / Longitude as numeric
for coord in ("LATITUDE", "LONGITUDE"):
     if coord in df.columns:
          df[coord] = pd.to_numeric(df[coord], errors="coerce")
     else:
          df[coord] = np.nan

# Parse ALL_VEHICLE_TYPES (which may be a string representation of a list) and create a flattened column
def parse_vehicle_list(v):
     if pd.isna(v):
          return []
     # If it's already a Python list object (rare in CSV), handle
     if isinstance(v, list):
          return [str(x).strip() for x in v if str(x).strip()]
     s = str(v).strip()
     # Try literal_eval if it's like "['SUV/Station Wagon', 'Sedan']"
     try:
          parsed = ast.literal_eval(s)
          if isinstance(parsed, (list, tuple)):
               return [str(x).strip() for x in parsed if str(x).strip()]
     except Exception:
          # fallback: comma-separated
          parts = [p.strip() for p in s.split(",") if p.strip()]
          return parts
     return []

df["VEHICLE_TYPES_LIST"] = df.get("ALL_VEHICLE_TYPES", "").apply(parse_vehicle_list)

# Expand vehicle types per row into a flat list column for easier counting
all_vehicle_types_flat = [vt for sub in df["VEHICLE_TYPES_LIST"] for vt in sub]
vehicle_type_counts = pd.Series(all_vehicle_types_flat).value_counts()
# Top 10 vehicle types for charts / heatmap combos
TOP_VEHICLE_TYPES = vehicle_type_counts.head(10).index.tolist()

# Parse contributing factors (all contributing factors column may be a list or string)
def parse_factor_list(v):
     if pd.isna(v):
          return []
     if isinstance(v, list):
          return [str(x).strip() for x in v if str(x).strip()]
     s = str(v).strip()
     try:
          parsed = ast.literal_eval(s)
          if isinstance(parsed, (list, tuple)):
               return [str(x).strip() for x in parsed if str(x).strip()]
     except Exception:
          parts = [p.strip() for p in s.split(",") if p.strip()]
          return parts
     return []

# Try to handle both ALL_CONTRIBUTING_FACTORS and ALL_CONTRIBUTING_FACTORS_STR
if "ALL_CONTRIBUTING_FACTORS" in df.columns:
     df["FACTORS_LIST"] = df["ALL_CONTRIBUTING_FACTORS"].apply(parse_factor_list)
elif "ALL_CONTRIBUTING_FACTORS_STR" in df.columns:
     df["FACTORS_LIST"] = df["ALL_CONTRIBUTING_FACTORS_STR"].apply(parse_factor_list)
else:
     # fallback to specific columns if provided
     parts = []
     for i in range(1, 4):
          c = f"CONTRIBUTING FACTOR VEHICLE {i}"
          if c in df.columns:
               parts.append(df[c].fillna("").astype(str))
     if parts:
          df["FACTORS_LIST"] = (pd.Series([";".join(x) for x in zip(*parts)]) if parts else pd.Series([[]]*len(df))).apply(
               lambda s: parse_factor_list(s))
     else:
          df["FACTORS_LIST"] = [[] for _ in range(len(df))]

all_factors_flat = [f for sub in df["FACTORS_LIST"] for f in sub]
factor_counts = pd.Series(all_factors_flat).value_counts()
TOP_FACTORS = factor_counts.head(10).index.tolist()

# PERSON_TYPE (type of persons involved)
if "PERSON_TYPE" not in df.columns and "PERSON_TYPE" in df.columns:
     pass
# ensure PERSON_TYPE column exists
if "PERSON_TYPE" not in df.columns:
     if "PERSON_TYPE" in df.columns:
          df["PERSON_TYPE"] = df["PERSON_TYPE"]
     else:
          df["PERSON_TYPE"] = df.get("PERSON_TYPE", "Unknown").fillna("Unknown")

# POSITION_IN_VEHICLE_CLEAN is provided in dataset per your list, ensure it's present
if "POSITION_IN_VEHICLE_CLEAN" not in df.columns:
     df["POSITION_IN_VEHICLE_CLEAN"] = df.get("POSITION_IN_VEHICLE_CLEAN", "").fillna("Unknown")

# Ensure other person-related columns exist (for new plots)
for col in ["PERSON_AGE", "PERSON_SEX", "BODILY_INJURY", "SAFETY_EQUIPMENT", "EMOTIONAL_STATUS", "UNIQUE_ID", "EJECTION", "ZIP CODE", "PERSON_INJURY"]:
    if col not in df.columns:
        # Create a placeholder column if not found (assuming person-level data is in the merged set)
        if col == "UNIQUE_ID":
            df[col] = df.index + 1
        elif col == "PERSON_AGE":
            df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce').fillna(0).astype(int) # Coerce age to int, fill missing/bad with 0
        elif col in ["EJECTION", "ZIP CODE", "PERSON_INJURY"]:
             df[col] = df.get(col, "Unknown").fillna("Unknown")
        else:
            df[col] = df.get(col, "Unknown").fillna("Unknown")

# Ensure additional columns exist
for col in ["COMPLAINT", "VEHICLE TYPE CODE 1", "CONTRIBUTING FACTOR VEHICLE 1"]:
    if col not in df.columns:
        df[col] = "Unknown"

# Small helper to add jitter to lat/lon to separate overlapping points
def jitter_coords(series, scale=0.0006):
     # scale tuned for city-level jitter
     return series + np.random.normal(loc=0, scale=scale, size=series.shape)

# Define consistent borough colors with proper capitalization
BOROUGH_COLORS = {
    'Manhattan': '#2ECC71',  # Green
    'Brooklyn': '#E74C3C',   # Red
    'Queens': '#3498DB',     # Blue
    'Bronx': '#F39C12',      # Orange
    'Staten Island': '#9B59B6', # Purple
    'Unknown': '#95A5A6'     # Gray
}

# ------------------------------------------------------------------
# Create Enhanced Dash app layout
# ------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# Year slider marks
min_year = int(df["YEAR"].min()) if not df["YEAR"].isna().all() else 2010
max_year = int(df["YEAR"].max()) if not df["YEAR"].isna().all() else pd.Timestamp.now().year
year_marks = {y: str(y) for y in range(min_year, max_year + 1)}

app.layout = dbc.Container([
    # Header with Summary
    dbc.Row([
        dbc.Col([
            html.H1("💥 NYC Crash Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
            html.Div(id="summary_text",
                    className="alert text-center",
                    style={'fontSize': '18px', 'fontWeight': 'bold', 'backgroundColor': '#FF8DA1', 'color': 'white', 'border': 'none'})
        ])
    ], className="mb-4"),

    # Interactive Control Panel
    dbc.Card([
        dbc.CardHeader(
            html.H4("📊 Control Panel", className="mb-0", style={'color': '#ffffff'}),
            style={'backgroundColor': '#FF8DA1'}
        ),
        dbc.CardBody([
         # Year Range Slider - Full Width
            dbc.Row([
                dbc.Col([
                    html.Label("Year Range", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dcc.RangeSlider(
                        id="year_slider",
                        min=min_year,
                        max=max_year,
                        value=[min_year, max_year],
                        marks={y: {'label': str(y), 'style': {'color': '#ffffff'}} for y in range(min_year, max_year + 1)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        step=1,
                        allowCross=False
                    ),
                ], width=12),
            ], className="mb-4"),

            # Filters Row 1
            dbc.Row([
                dbc.Col([
                    html.Label("Borough", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="borough_filter",
                        options=[{"label": b, "value": b} for b in sorted(df["BOROUGH"].dropna().unique())],
                        multi=True,
                        placeholder="All Boroughs",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Vehicle Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="vehicle_filter",
                        options=[{"label": v, "value": v}
                                for v in sorted({vt for sub in df["VEHICLE_TYPES_LIST"] for vt in sub})],
                        multi=True,
                        placeholder="All Vehicle Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Contributing Factor", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="factor_filter",
                        options=[{"label": f, "value": f}
                                for f in sorted({f for sub in df["FACTORS_LIST"] for f in sub})],
                        multi=True,
                        placeholder="All Factors",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
                dbc.Col([
                    html.Label("Person Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="person_type_filter",
                        options=[{"label": v, "value": v} for v in sorted(df["PERSON_TYPE"].dropna().unique())],
                        multi=True,
                        placeholder="All Person Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=3),
            ], className="mb-3"),

            # Filters Row 2
            dbc.Row([
                dbc.Col([
                    html.Label("Injury Type", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="injury_filter",
                        options=[{"label": i, "value": i} for i in sorted(df["PERSON_INJURY"].dropna().unique())],
                        multi=True,
                        placeholder="All Injury Types",
                        style={'backgroundColor': '#FFE6E6', 'border': '1px solid #FFB6C1'}
                    )
                ], width=8),
                dbc.Col([
                    html.Label("Clear Filters", style={'color': '#ffffff', 'fontWeight': 'bold'}),
                    dbc.Button("🗑️ Clear All Filters",
                              id="clear_filters_btn",
                              color="warning",
                              size="md",
                              className="w-100",
                              style={
                                  'backgroundColor': '#FF6B6B',
                                  'border': 'none',
                                  'fontWeight': 'bold',
                                  'color': 'white'
                              })
                ], width=4),
            ], className="mb-4"),

            # Search Section
            dbc.Row([
                dbc.Col([
                    html.Label("🔍 Advanced Search", style={'color': '#ffffff', 'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.Input(
                        id="search_input",
                        placeholder="Try: 'queens 2019 to 2022 bicycle female pedestrian'...",
                        type="text",
                        style={
                            'backgroundColor': '#FFE6E6',
                            'border': '2px solid #FF8DA1',
                            'color': '#333',
                            'fontSize': '14px',
                            'padding': '12px'
                        }
                    ),
                    dbc.FormText(
                        "Search by borough, year, vehicle type, gender, injury type",
                        style={'color': '#ffffff', 'fontWeight': 'bold'}
                    )
                ], width=12),
            ], className="mb-4"),

            # Update Button
            dbc.Row([
                dbc.Col([
                    dbc.Button("🔄 Update Dashboard",
                              id="generate_btn",
                              color="primary",
                              size="lg",
                              className="w-100",
                              style={'backgroundColor': '#FF8DA1', 'border': 'none', 'fontWeight': 'bold'})
                ], width=12),
            ]),
        ], style={'backgroundColor': '#add8e6'})
    ], className="mb-4", style={'border': '2px solid #FF8DA1'}),

    # Tabbed Interface for Organized Content
    dbc.Tabs([
        # Tab 1: Crash Overview & Geography
        dbc.Tab([
            html.Br(),
            # Crash Map - Full Width
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📍 Crash Locations Map", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="map_chart", style={'height': '500px'})
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            # Crash Trends - Full Width
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Crash Trends Over Time", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_year")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏙️ Crashes by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="borough_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💥 Injuries by Borough", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injuries_by_borough")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),
        ], label="🗺️ Crash Geography", tab_id="tab-1"),

        # Tab 2: Vehicle & Factor Analysis
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔧 Contributing Factors", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="crashes_by_factor")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔥 Vehicle vs Factor Heatmap", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_factor_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🏎️ Vehicle Type Trends", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="vehicle_trend_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="🏎️ Vehicles & Factors", tab_id="tab-2"),

        # Tab 3: Person & Injury Analysis
        dbc.Tab([
            html.Br(),
            # First row with Safety Equipment and Injury Types side by side
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🛡️ Safety Equipment", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="safety_equipment")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚑 Injury Types", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injury_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            # Second row with the other three charts
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🎭 Emotional State", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="emotional_state")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚪 Ejection Status", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="ejection_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("💺 Position in Vehicle", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="position_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
            ], className="mb-4"),

            # Third row with remaining charts
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("👥 Person Types Over Time", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="injuries_by_person_type")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📋 Top Complaints", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="complaint_chart")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),
        ], label="👥 People & Injuries", tab_id="tab-3"),

        # Tab 4: Demographics & Statistics
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📊 Age Distribution", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="age_distribution_hist")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=8),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🚻 Gender Distribution", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    dcc.Graph(id="gender_distribution")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=4),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Real-time Statistics", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.Div(id="live_stats", className="text-center")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="📈 Demographics", tab_id="tab-4"),

        # NEW TAB 5: Advanced Analytics
        dbc.Tab([
            html.Br(),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🔥 Crash Hotspot Clustering", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Identifies geographic clusters of high crash frequency using machine learning",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="hotspot_cluster_map")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📊 Risk Correlation Matrix", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Shows relationships between different risk factors and crash outcomes",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="correlation_heatmap")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🕒 Temporal Risk Patterns", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Reveals peak crash times by day of week and hour for targeted interventions",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="temporal_patterns")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("🎯 Severity Prediction Factors", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Analyzes which boroughs and factors lead to the most severe crash outcomes",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="severity_factors")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=6),
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("📈 Spatial Risk Density", style={'color': '#2C3E50', 'fontWeight': 'bold'}),
                    html.P("Heatmap showing geographic concentration of severe crashes and high-risk zones",
                          style={'color': '#666', 'fontSize': '14px'}),
                    dcc.Graph(id="risk_density_map")
                ]), style={'backgroundColor': '#FFE6E6', 'border': '2px solid #FF8DA1'}), md=12),
            ], className="mb-4"),
        ], label="🔬 Advanced Analytics", tab_id="tab-5"),
    ], id="tabs", active_tab="tab-1",
       style={'marginTop': '20px'},
       className="custom-tabs"),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(style={'borderColor': '#FF8DA1'}),
            html.P("NYC Crash Analysis Dashboard | Built with Dash & Plotly",
                  className="text-center",
                  style={'color': '#ffffff', 'fontWeight': 'bold'})
        ])
    ], className="mt-4")

], fluid=True, style={'backgroundColor': '#cee8f0', 'minHeight': '100vh', 'padding': '20px'})

# Add custom CSS for tab styling AND slider
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* SLIDER STYLING */
            .rc-slider-track {
                background-color: #fc94af !important;
            }
            .rc-slider-rail {
                background-color: #FFC0CB !important;
            }
            .rc-slider-handle {
                background-color: #fc94af !important;
                border: 2px solid white !important;
            }

            /* TAB STYLING */
            .custom-tabs .nav-link {
                background-color: #FF8DA1 !important;
                color: white !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
                margin-right: 5px;
            }
            .custom-tabs .nav-link.active {
                background-color: white !important;
                color: #FF8DA1 !important;
                border: 1px solid #FF8DA1 !important;
                font-weight: bold !important;
            }
            .custom-tabs .nav-link:hover {
                background-color: #FF85A1 !important;
                color: white !important;
            }
            .custom-tabs .nav-link.active:hover {
                background-color: white !important;
                color: #FF8DA1 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# -------------------------
# Helper: parse search query - SIMPLIFIED VERSION (no contributing factors in search)
# -------------------------
def parse_search_query(q):
    q = (q or "").lower().strip()
    found = {}

    # Extract years and year ranges
    year_pattern = r'\b(20\d{2})\b'
    years_found = re.findall(year_pattern, q)
    if years_found:
        years = sorted([int(y) for y in years_found])
        if len(years) >= 2:
            found["year_range"] = [years[0], years[-1]]
        else:
            found["year"] = years[0]

    # Borough detection
    borough_keywords = {
        'manhattan': 'Manhattan',
        'brooklyn': 'Brooklyn',
        'queens': 'Queens',
        'bronx': 'Bronx',
        'staten': 'Staten Island',
        'staten island': 'Staten Island'
    }
    for keyword, borough in borough_keywords.items():
        if keyword in q:
            found["borough"] = [borough]
            break

    # Vehicle type detection
    vehicle_keywords = {
        'suv': 'SUV/Station Wagon',
        'station wagon': 'SUV/Station Wagon',
        'sedan': 'Sedan',
        'bicycle': 'Bicycle',
        'bike': 'Bicycle',
        'ambulance': 'Ambulance',
        'bus': 'Bus',
        'motorcycle': 'Motorcycle',
        'pickup': 'Pickup Truck',
        'pickup truck': 'Pickup Truck',
        'taxi': 'Taxi',
        'truck': 'Truck/Commercial',
        'commercial': 'Truck/Commercial',
        'van': 'Van',
        'pedicab': 'Pedicab'
    }
    vehicle_matches = []
