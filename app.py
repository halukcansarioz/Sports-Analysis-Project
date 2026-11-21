import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

st.set_page_config(page_title="NBA Analytics & Fantasy Predictor", page_icon="🏀", layout="wide")
PROFILES = {
    "Guard (Playmaking Scorer)": {
        "MIN": 35.0, "PTS": 24.5, "REB": 4.8, "AST": 8.6,
        "STL": 1.5, "BLK": 0.4, "TOV": 3.1, "FG%": 47.5,
    },
    "Wing (Two-Way Forward)": {
        "MIN": 34.5, "PTS": 26.0, "REB": 7.2, "AST": 4.5,
        "STL": 1.3, "BLK": 0.8, "TOV": 2.4, "FG%": 49.0,
    },
    "Center (Rim Protector & Hub)": {
        "MIN": 33.0, "PTS": 22.5, "REB": 12.4, "AST": 4.2,
        "STL": 0.9, "BLK": 2.3, "TOV": 2.6, "FG%": 58.0,
    },
}

FEATURES = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG%"]

def calc_standard_fantasy(pts, reb, ast, stl, blk, tov):
    return pts * 1.0 + reb * 1.2 + ast * 1.5 + stl * 3.0 + blk * 3.0 - tov * 1.0

@st.cache_resource
def train_fantasy_model():
    rng = np.random.default_rng(42)
    n = 400
    mins = rng.uniform(18, 40, n)
    fg_pct = rng.uniform(38, 65, n)
    pts = mins * rng.uniform(0.45, 0.85, n) * (fg_pct / 48.0)
    reb = mins * rng.uniform(0.12, 0.38, n)
    ast = mins * rng.uniform(0.08, 0.28, n)
    stl = rng.uniform(0.3, 2.2, n)
    blk = rng.uniform(0.1, 2.6, n)
    tov = rng.uniform(1.0, 4.2, n)
    X = np.column_stack([mins, pts, reb, ast, stl, blk, tov, fg_pct])
    base = calc_standard_fantasy(pts, reb, ast, stl, blk, tov)
    y = base + 0.08 * (fg_pct - 47.0) + 0.05 * (mins - 30.0) + rng.normal(0, 1.8, n)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model
  
st.title("🏀 NBA Performance Analytics & Fantasy Points Predictor")
st.caption("Interactive demo comparing the NBA standard fantasy scoring formula with a Ridge regression model.")

model = train_fantasy_model()

with st.sidebar:
    st.header("Player Archetype & Recent Averages")
    archetype = st.selectbox("Select Player Profile", list(PROFILES.keys()))
    defaults = PROFILES[archetype]

    min_val = st.slider("Minutes (MIN)", 10.0, 44.0, defaults["MIN"], 0.5)
    pts_val = st.slider("Points (PTS)", 0.0, 45.0, defaults["PTS"], 0.5)
    reb_val = st.slider("Rebounds (REB)", 0.0, 20.0, defaults["REB"], 0.2)
    ast_val = st.slider("Assists (AST)", 0.0, 16.0, defaults["AST"], 0.2)
    stl_val = st.slider("Steals (STL)", 0.0, 4.0, defaults["STL"], 0.1)
    blk_val = st.slider("Blocks (BLK)", 0.0, 5.0, defaults["BLK"], 0.1)
    tov_val = st.slider("Turnovers (TOV)", 0.0, 7.0, defaults["TOV"], 0.1)
    fg_val = st.slider("Field Goal % (FG%)", 30.0, 75.0, defaults["FG%"], 0.5)

std_score = calc_standard_fantasy(pts_val, reb_val, ast_val, stl_val, blk_val, tov_val)
X_input = np.array([[min_val, pts_val, reb_val, ast_val, stl_val, blk_val, tov_val, fg_val]])
reg_score = float(model.predict(X_input)[0])
delta = reg_score - std_score

col1, col2, col3 = st.columns(3)
col1.metric("NBA Standard Fantasy Score", f"{std_score:.2f} pts")
col2.metric("Regression Model Prediction", f"{reg_score:.2f} pts", delta=f"{delta:+.2f} vs formula")
col3.metric("Selected Archetype", archetype.split(" (")[0])

st.divider()
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Last 10 Games Trend & Next-Game Projection")
    rng_trend = np.random.default_rng(int((pts_val + reb_val + ast_val) * 10))
    games = [f"G-{i}" for i in range(10, 0, -1)] + ["Next (Proj)"]
    hist_scores = std_score + rng_trend.normal(0, 4.2, 10)
    trend_df = pd.DataFrame({
        "NBA Standard": np.append(hist_scores, std_score),
        "Regression Pred": np.append(hist_scores + delta * 0.5, reg_score),
    }, index=games)
    st.line_chart(trend_df)

with right_col:
    st.subheader("Stat Contribution Breakdown (Formula)")
    contribs = {
        "PTS (x1.0)": pts_val * 1.0,
        "REB (x1.2)": reb_val * 1.2,
        "AST (x1.5)": ast_val * 1.5,
        "STL (x3.0)": stl_val * 3.0,
        "BLK (x3.0)": blk_val * 3.0,
        "TOV (x-1.0)": -tov_val * 1.0,
    }
    contrib_df = pd.DataFrame({"Fantasy Points": contribs.values()}, index=contribs.keys())
    st.bar_chart(contrib_df)

with st.expander("Model Details & Scoring Formula"):
    st.markdown("**NBA Standard Formula:** `PTS*1.0 + REB*1.2 + AST*1.5 + STL*3.0 + BLK*3.0 - TOV*1.0`")
    coef_df = pd.DataFrame({"Feature": FEATURES, "Ridge Coefficient": np.round(model.coef_, 3)})
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
