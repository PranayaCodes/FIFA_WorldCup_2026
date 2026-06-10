import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
    .stApp { background-color: #080c14; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }
    html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, sans-serif; color: #e2e8f0; }
    h1 { font-size: 2.4rem !important; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2 { font-size: 1.5rem !important; font-weight: 700 !important; }
    h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #94a3b8 !important; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1623 0%, #0a0f1a 100%) !important;
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
        color: #cbd5e1 !important; font-size: 0.95rem !important; padding: 8px 12px !important;
        border-radius: 8px !important; transition: all 0.2s; text-transform: none !important; letter-spacing: 0 !important;
    }
    .gold-card {
        background: linear-gradient(135deg, #13192b 0%, #0e1420 100%);
        border: 1px solid #2a3548; border-top: 3px solid #f59e0b;
        border-radius: 12px; padding: 20px; text-align: center; position: relative; overflow: hidden;
    }
    .gold-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 60px;
        background: linear-gradient(180deg, rgba(245,158,11,0.06) 0%, transparent 100%); pointer-events: none;
    }
    .gold-card .rank-icon { font-size: 2rem; margin-bottom: 8px; }
    .gold-card .team-name { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin: 6px 0; }
    .gold-card .prob-value { font-size: 2rem; font-weight: 800; color: #f59e0b; line-height: 1; }
    .gold-card .prob-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .section-header {
        display: flex; align-items: center; gap: 10px; margin: 2rem 0 1rem;
        padding-bottom: 10px; border-bottom: 1px solid #1e293b;
    }
    .section-header i { color: #f59e0b; font-size: 1.2rem; }
    .section-header span { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; }
    .vs-badge {
        background: linear-gradient(135deg, #1a2235, #131929); border: 2px solid #f59e0b;
        border-radius: 50%; width: 60px; height: 60px; display: flex; align-items: center;
        justify-content: center; font-size: 1rem; font-weight: 800; color: #f59e0b; margin: auto;
    }
    .result-bar { background: #1e293b; border-radius: 6px; height: 10px; overflow: hidden; margin: 8px 0; }
    .result-bar-fill { height: 100%; border-radius: 6px; }
    .group-card { background: #0f1623; border: 1px solid #1e293b; border-radius: 10px; padding: 14px 16px; margin-bottom: 14px; }
    .group-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: #f59e0b; margin-bottom: 10px; }
    .group-team-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #1a2235; font-size: 0.88rem; }
    .group-team-row:last-child { border-bottom: none; }
    .team-prob-badge { font-size: 0.72rem; font-weight: 700; color: #f59e0b; background: rgba(245,158,11,0.1); border-radius: 20px; padding: 2px 8px; }
    .stat-row { display: flex; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid #1a2235; }
    .prob-card-home { background: linear-gradient(135deg, #0d2218, #0a1a12); border: 1px solid #166534; border-radius: 12px; padding: 24px 16px; text-align: center; }
    .prob-card-draw { background: linear-gradient(135deg, #1a1508, #130f05); border: 1px solid #854d0e; border-radius: 12px; padding: 24px 16px; text-align: center; }
    .prob-card-away { background: linear-gradient(135deg, #1a0808, #120505); border: 1px solid #991b1b; border-radius: 12px; padding: 24px 16px; text-align: center; }
    .prob-team { font-size: 0.88rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px; }
    .prob-big { font-size: 2.8rem; font-weight: 800; line-height: 1; }
    .prob-big.green { color: #4ade80; }
    .prob-big.gold { color: #f59e0b; }
    .prob-big.red { color: #f87171; }
    .prob-sub { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
    .mini-stat { background: #0f1623; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; text-align: center; }
    .mini-stat .val { font-size: 1.8rem; font-weight: 800; color: #f59e0b; }
    .mini-stat .lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .page-divider { border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0; }
    .stButton > button {
        background: linear-gradient(135deg, #b45309, #92400e) !important; color: #fef3c7 !important;
        border: none !important; border-radius: 10px !important; font-weight: 700 !important;
        font-size: 1rem !important; padding: 12px 24px !important; letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(180,83,9,0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #d97706, #b45309) !important;
        box-shadow: 0 6px 20px rgba(217,119,6,0.4) !important;
    }
    .stSelectbox > div > div { background: #0f1623 !important; border: 1px solid #2a3548 !important; border-radius: 10px !important; color: #f1f5f9 !important; }
    .stDataFrame { border: 1px solid #1e293b !important; border-radius: 10px !important; overflow: hidden; }
    .stAlert { border-radius: 10px !important; border-left: 4px solid #f59e0b !important; }
    .stProgress > div > div { background: linear-gradient(90deg, #b45309, #f59e0b) !important; border-radius: 4px !important; }
    .champion-banner {
        background: linear-gradient(135deg, #1a1205 0%, #0f0b03 100%); border: 2px solid #f59e0b;
        border-radius: 16px; padding: 32px 24px; text-align: center;
    }
    .champion-banner .trophy { font-size: 3.5rem; }
    .champion-banner .name { font-size: 2rem; font-weight: 800; color: #f59e0b; margin: 12px 0 4px; }
    .version-card {
        background: #0f1623; border: 1px solid #1e293b; border-left: 4px solid #f59e0b;
        border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
    }
    .version-card .ver-tag { font-size: 0.72rem; font-weight: 700; color: #f59e0b; text-transform: uppercase; letter-spacing: 1.5px; }
    .version-card .ver-title { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin: 4px 0; }
    .version-card .ver-acc { font-size: 1.4rem; font-weight: 800; color: #4ade80; }
    .version-card .ver-desc { font-size: 0.82rem; color: #64748b; margin-top: 6px; }
    .hero-sub { font-size: 0.9rem; color: #475569; margin-top: -8px; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    with open(os.path.join(BASE, 'Model_Training', 'xgb_model_v3.pkl'), 'rb') as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(BASE, 'Model_Training', 'label_encoder_v3.pkl'), 'rb') as f:
        le = pickle.load(f)
    return xgb_model, le

@st.cache_data
def load_data():
    predictions = pd.read_csv(os.path.join(BASE, 'Monte_Carlo_Simulation', 'predictions_final.csv'))
    features_df = pd.read_csv(os.path.join(BASE, 'Feature_Engineering', 'features_v3.csv'))
    elo = pd.read_csv(os.path.join(BASE, 'Cleaned_Data', 'elo_cleaned.csv'))
    player_strength = pd.read_csv(os.path.join(BASE, 'Cleaned_Data', 'player_strength.csv'))
    star_elo = pd.read_csv(os.path.join(BASE, 'Cleaned_Data', 'star_player_elo.csv'))
    squad_age = pd.read_csv(os.path.join(BASE, 'Cleaned_Data', 'squad_age.csv'))
    elo_conf = pd.read_csv(os.path.join(BASE, 'Cleaned_Data', 'confederation_strength.csv'))
    return predictions, features_df, elo, player_strength, star_elo, squad_age, elo_conf

xgb_model, le = load_models()
predictions, features_df, elo, player_strength, star_elo, squad_age, elo_conf = load_data()

elo_lookup = elo.set_index('team')['elo_rating'].to_dict()
elo_wr_lookup = elo.set_index('team')['elo_win_rate'].to_dict()
star_elo_lookup = star_elo.set_index('team')['star_player_elo'].to_dict()
age_lookup = squad_age.set_index('team')['age_score'].to_dict()
conf_lookup = elo_conf.set_index('team')['conf_strength'].to_dict()
ps = player_strength.set_index('team')

def get_player_stat(team, col, default=0):
    try:
        val = ps.loc[team, col]
        return val if not pd.isna(val) else default
    except:
        return default

wc_groups = {
    'A': ['Mexico', 'South Korea', 'South Africa', 'Czech Republic'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Colombia', 'Uzbekistan'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama']
}
wc_teams = [team for group in wc_groups.values() for team in group]
host_nations = ['United States', 'Canada', 'Mexico']
FAVORITES = ['Spain', 'France', 'Portugal', 'England', 'Argentina', 'Brazil', 'Germany']

def get_team_stats(team, df, n=20):
    team_matches = df[(df['home_team'] == team) | (df['away_team'] == team)].tail(n)
    if len(team_matches) == 0:
        return {'form': 0.5, 'avg_scored': 1.2, 'avg_conceded': 1.2}
    wins, scored, conceded = 0, [], []
    for _, row in team_matches.iterrows():
        if row['home_team'] == team:
            scored.append(row['home_avg_scored']); conceded.append(row['home_avg_conceded'])
            if row['result'] == 1: wins += 1
        else:
            scored.append(row['away_avg_scored']); conceded.append(row['away_avg_conceded'])
            if row['result'] == -1: wins += 1
    return {'form': wins / len(team_matches), 'avg_scored': np.mean(scored), 'avg_conceded': np.mean(conceded)}

team_stats = {team: get_team_stats(team, features_df) for team in wc_teams}

def get_h2h(home, away):
    h2h = features_df[
        (((features_df['home_team'] == home) & (features_df['away_team'] == away)) |
         ((features_df['home_team'] == away) & (features_df['away_team'] == home)))
    ].tail(10)
    if len(h2h) == 0: return 0.5, 0
    home_wins = 0
    for _, row in h2h.iterrows():
        if row['home_team'] == home and row['result'] == 1: home_wins += 1
        elif row['away_team'] == home and row['result'] == -1: home_wins += 1
    return home_wins / len(h2h), len(h2h)

def predict_match(home_team, away_team):
    home = team_stats[home_team]; away = team_stats[away_team]
    home_elo = elo_lookup.get(home_team, 1500); away_elo = elo_lookup.get(away_team, 1500)
    home_star = star_elo_lookup.get(home_team, 1500); away_star = star_elo_lookup.get(away_team, 1500)
    h2h_rate, _ = get_h2h(home_team, away_team)
    feat = np.array([[
        home['form'], away['form'], home['form'] - away['form'],
        home['avg_scored'], home['avg_conceded'], away['avg_scored'], away['avg_conceded'],
        home['avg_scored'] - away['avg_scored'], 1,
        home_elo, away_elo, home_elo - away_elo,
        elo_wr_lookup.get(home_team, 0.5), elo_wr_lookup.get(away_team, 0.5),
        home_star, away_star, home_star - away_star,
        get_player_stat(home_team, 'attack_goals'), get_player_stat(away_team, 'attack_goals'),
        get_player_stat(home_team, 'attack_goals') - get_player_stat(away_team, 'attack_goals'),
        get_player_stat(home_team, 'def_tackles'), get_player_stat(away_team, 'def_tackles'),
        get_player_stat(home_team, 'def_tackles') - get_player_stat(away_team, 'def_tackles'),
        get_player_stat(home_team, 'mid_tackles'), get_player_stat(away_team, 'mid_tackles'),
        h2h_rate, 0.0,
        age_lookup.get(home_team, 0.5), age_lookup.get(away_team, 0.5),
        conf_lookup.get(home_team, 0.7), conf_lookup.get(away_team, 0.7),
        conf_lookup.get(home_team, 0.7) - conf_lookup.get(away_team, 0.7)
    ]])
    return xgb_model.predict_proba(feat)[0]

def predict_score_poisson(home_team, away_team):
    home = team_stats[home_team]; away = team_stats[away_team]
    home_attack = home['avg_scored'] * (1 + get_player_stat(home_team, 'attack_goals') * 0.05)
    away_attack = away['avg_scored'] * (1 + get_player_stat(away_team, 'attack_goals') * 0.05)
    home_defense = 1 / (1 + get_player_stat(home_team, 'def_tackles') * 0.01)
    away_defense = 1 / (1 + get_player_stat(away_team, 'def_tackles') * 0.01)
    elo_factor = (elo_lookup.get(home_team, 1500) - elo_lookup.get(away_team, 1500)) / 1000
    star_factor = (star_elo_lookup.get(home_team, 1500) - star_elo_lookup.get(away_team, 1500)) / 5000
    conf_factor = (conf_lookup.get(home_team, 0.7) - conf_lookup.get(away_team, 0.7)) * 0.3
    host_boost = 0.15 if home_team in host_nations else 0
    home_xg = max(0.3, home_attack * away_defense * (1 + elo_factor + star_factor + conf_factor + host_boost))
    away_xg = max(0.3, away_attack * home_defense * (1 - elo_factor - star_factor - conf_factor))
    return np.random.poisson(home_xg), np.random.poisson(away_xg)

def simulate_group(group_teams):
    points = {t: 0 for t in group_teams}; gd = {t: 0 for t in group_teams}; gf = {t: 0 for t in group_teams}
    for i in range(len(group_teams)):
        for j in range(i+1, len(group_teams)):
            hg, ag = predict_score_poisson(group_teams[i], group_teams[j])
            gf[group_teams[i]] += hg; gf[group_teams[j]] += ag
            gd[group_teams[i]] += hg - ag; gd[group_teams[j]] += ag - hg
            if hg > ag: points[group_teams[i]] += 3
            elif ag > hg: points[group_teams[j]] += 3
            else: points[group_teams[i]] += 1; points[group_teams[j]] += 1
    return sorted(group_teams, key=lambda t: (points[t], gd[t], gf[t]), reverse=True), points, gd

def simulate_knockout(teams):
    remaining = teams.copy()
    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining) - 1, 2):
            hg, ag = predict_score_poisson(remaining[i], remaining[i+1])
            if hg > ag: next_round.append(remaining[i])
            elif ag > hg: next_round.append(remaining[i+1])
            else: next_round.append(remaining[i] if np.random.random() > 0.5 else remaining[i+1])
        if len(remaining) % 2 == 1: next_round.append(remaining[-1])
        remaining = next_round
    return remaining[0]

def run_simulation(n=1000):
    win_counts = {team: 0 for team in wc_teams}
    for _ in range(n):
        group_winners, group_runners_up, all_third = [], [], []
        for group_teams in wc_groups.values():
            standings, points, gd = simulate_group(group_teams)
            group_winners.append(standings[0]); group_runners_up.append(standings[1])
            all_third.append((standings[2], points[standings[2]], gd[standings[2]]))
        all_third.sort(key=lambda x: (x[1], x[2]), reverse=True)
        r32 = group_winners + group_runners_up + [t[0] for t in all_third[:8]]
        np.random.shuffle(r32)
        win_counts[simulate_knockout(r32)] += 1
    return {team: (count / n) * 100 for team, count in win_counts.items()}

def make_dark_chart(figsize=(11, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#080c14'); ax.set_facecolor('#0a0f1a')
    ax.tick_params(colors='#64748b', labelsize=10)
    for spine in ax.spines.values(): spine.set_color('#1e293b')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color='#1e293b', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='padding: 20px 10px 10px; text-align: center;'>
    <div style='font-size: 3rem; margin-bottom: 4px;'>🏆</div>
    <div style='font-size: 1.3rem; font-weight: 800; color: #f59e0b; letter-spacing: -0.3px;'>WC 2026</div>
    <div style='font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 2px; margin-top: 2px;'>Predictor</div>
</div>
<hr style='border-color:#1e293b; margin: 10px 0 18px;'>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "Tournament Predictions", "Match Predictor", "Run Simulation", "Model Info"
])

st.sidebar.markdown("<hr style='border-color:#1e293b; margin: 18px 0 14px;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='padding: 0 4px;'>
  <div style='font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:#334155; margin-bottom:12px;'>Model Stats</div>
  <div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2235;'>
    <span style='font-size:0.82rem; color:#64748b;'><i class='fas fa-robot' style='color:#f59e0b; margin-right:6px;'></i>Model</span>
    <span style='font-size:0.82rem; font-weight:700; color:#f1f5f9;'>XGBoost V3</span>
  </div>
  <div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2235;'>
    <span style='font-size:0.82rem; color:#64748b;'><i class='fas fa-bullseye' style='color:#4ade80; margin-right:6px;'></i>Accuracy</span>
    <span style='font-size:0.82rem; font-weight:700; color:#4ade80;'>56%</span>
  </div>
  <div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #1a2235;'>
    <span style='font-size:0.82rem; color:#64748b;'><i class='fas fa-layer-group' style='color:#60a5fa; margin-right:6px;'></i>Features</span>
    <span style='font-size:0.82rem; font-weight:700; color:#f1f5f9;'>32</span>
  </div>
  <div style='display:flex; justify-content:space-between; padding:8px 0;'>
    <span style='font-size:0.82rem; color:#64748b;'><i class='fas fa-dice' style='color:#c084fc; margin-right:6px;'></i>Simulations</span>
    <span style='font-size:0.82rem; font-weight:700; color:#f1f5f9;'>10,000</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── PAGE 1: TOURNAMENT PREDICTIONS ───────────────────────────────────────────
if page == "Tournament Predictions":
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; color:#f59e0b;'>
            <i class='fas fa-trophy' style='margin-right:6px;'></i>FIFA World Cup 2026
        </span>
    </div>
    <h1 style='color:#f1f5f9; margin:0 0 6px;'>Winner Predictions</h1>
    <p class='hero-sub'>Based on 10,000 Monte Carlo simulations — ELO ratings, player strength, H2H records & squad age</p>
    """, unsafe_allow_html=True)

    top5 = predictions.head(5)
    medal_html = [
        ("<i class='fas fa-medal' style='color:#FFD700;'></i>", "#FFD700"),
        ("<i class='fas fa-medal' style='color:#C0C0C0;'></i>", "#C0C0C0"),
        ("<i class='fas fa-medal' style='color:#CD7F32;'></i>", "#CD7F32"),
        ("<i class='fas fa-star' style='color:#60a5fa;'></i>", "#60a5fa"),
        ("<i class='fas fa-star' style='color:#60a5fa;'></i>", "#60a5fa"),
    ]
    cols = st.columns(5)
    for i, (col, (_, row)) in enumerate(zip(cols, top5.iterrows())):
        icon, color = medal_html[i]
        with col:
            st.markdown(f"""
            <div class='gold-card'>
                <div class='rank-icon'>{icon}</div>
                <div class='team-name'>{row['Team']}</div>
                <div class='prob-value' style='color:{color};'>{row['Win Probability (%)']:.1f}%</div>
                <div class='prob-label'>win probability</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.markdown("<div class='section-header'><i class='fas fa-chart-bar'></i><span>Win Probability — Top 20</span></div>", unsafe_allow_html=True)
        top20 = predictions.head(20).sort_values('Win Probability (%)', ascending=True)
        bar_colors = ['#f59e0b' if t in FAVORITES else '#1e40af' for t in top20['Team']]
        fig, ax = make_dark_chart(figsize=(11, 8))
        bars = ax.barh(top20['Team'], top20['Win Probability (%)'], color=bar_colors, height=0.65, edgecolor='none')
        for bar, val in zip(bars, top20['Win Probability (%)']):
            ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', ha='left', color='#94a3b8', fontsize=9)
        ax.set_xlabel('Win Probability (%)', color='#64748b', fontsize=10)
        ax.tick_params(axis='y', colors='#cbd5e1', labelsize=10)
        ax.tick_params(axis='x', colors='#475569', labelsize=9)
        ax.set_title('FIFA World Cup 2026 — Predicted Win Probabilities', color='#94a3b8', fontsize=11, pad=14, loc='left')
        gold_patch = mpatches.Patch(color='#f59e0b', label='Favorites')
        blue_patch = mpatches.Patch(color='#1e40af', label='Others')
        ax.legend(handles=[gold_patch, blue_patch], facecolor='#0a0f1a', edgecolor='#1e293b', labelcolor='#94a3b8', fontsize=9, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("<div class='section-header'><i class='fas fa-list-ol'></i><span>Full Rankings</span></div>", unsafe_allow_html=True)
        display_df = predictions[['Rank', 'Team', 'Win Probability (%)']].copy()
        display_df['Win Probability (%)'] = display_df['Win Probability (%)'].round(2).astype(str) + '%'
        st.dataframe(display_df, height=580, use_container_width=True, hide_index=True)

    st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><i class='fas fa-globe-americas'></i><span>Official 2026 World Cup Groups</span></div>", unsafe_allow_html=True)
    groups_list = list(wc_groups.items())
    gcols = st.columns(4)
    for idx, (g, teams) in enumerate(groups_list):
        with gcols[idx % 4]:
            team_rows = ""
            for team in teams:
                prob = predictions[predictions['Team'] == team]['Win Probability (%)'].values
                prob_str = f"{prob[0]:.1f}%" if len(prob) > 0 else "—"
                team_rows += f"<div class='group-team-row'><span style='color:#cbd5e1;'>{team}</span><span class='team-prob-badge'>{prob_str}</span></div>"
            st.markdown(f"<div class='group-card'><div class='group-title'>Group {g}</div>{team_rows}</div>", unsafe_allow_html=True)

# ── PAGE 2: MATCH PREDICTOR ───────────────────────────────────────────────────
elif page == "Match Predictor":
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; color:#f59e0b;'>
            <i class='fas fa-futbol' style='margin-right:6px;'></i>Head-to-Head
        </span>
    </div>
    <h1 style='color:#f1f5f9; margin:0 0 6px;'>Match Predictor</h1>
    <p class='hero-sub'>Select any two teams to get a full ML-powered match breakdown</p>
    <hr class='page-divider'>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 2, 5])
    with col1:
        st.markdown("<p style='font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Home / Team 1</p>", unsafe_allow_html=True)
        home_team = st.selectbox("", sorted(wc_teams), index=sorted(wc_teams).index('England'), label_visibility="collapsed")
    with col2:
        st.markdown("<div style='height:52px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='vs-badge'>VS</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<p style='font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Away / Team 2</p>", unsafe_allow_html=True)
        away_options = [t for t in sorted(wc_teams) if t != home_team]
        away_team = st.selectbox("", away_options, index=away_options.index('France') if 'France' in away_options else 0, label_visibility="collapsed")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚽  Predict Match", use_container_width=True)

    if predict_btn:
        with st.spinner("Running model analysis..."):
            probs = predict_match(home_team, away_team)
            home_win_prob = probs[2] * 100
            draw_prob = probs[1] * 100
            away_win_prob = probs[0] * 100

        st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='prob-card-home'>
                <div class='prob-team'>{home_team}</div>
                <div class='prob-big green'>{home_win_prob:.1f}%</div>
                <div class='result-bar'><div class='result-bar-fill' style='width:{home_win_prob:.1f}%; background:#4ade80;'></div></div>
                <div class='prob-sub'>Win probability</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='prob-card-draw'>
                <div class='prob-team'>Draw</div>
                <div class='prob-big gold'>{draw_prob:.1f}%</div>
                <div class='result-bar'><div class='result-bar-fill' style='width:{draw_prob:.1f}%; background:#f59e0b;'></div></div>
                <div class='prob-sub'>Draw probability</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='prob-card-away'>
                <div class='prob-team'>{away_team}</div>
                <div class='prob-big red'>{away_win_prob:.1f}%</div>
                <div class='result-bar'><div class='result-bar-fill' style='width:{away_win_prob:.1f}%; background:#f87171;'></div></div>
                <div class='prob-sub'>Win probability</div></div>""", unsafe_allow_html=True)

        winner = home_team if home_win_prob > away_win_prob else away_team
        winner_prob = max(home_win_prob, away_win_prob)
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d1a0a,#0a1308); border:1px solid #166534;
                    border-radius:12px; padding:18px 24px; margin-top:16px;
                    display:flex; align-items:center; gap:16px;'>
            <i class='fas fa-trophy' style='color:#f59e0b; font-size:2rem;'></i>
            <div>
                <div style='font-size:0.72rem; color:#4ade80; text-transform:uppercase; letter-spacing:1.5px;'>Predicted Winner</div>
                <div style='font-size:1.4rem; font-weight:800; color:#f1f5f9; margin-top:2px;'>
                    {winner} <span style='color:#64748b; font-size:1rem; font-weight:400;'>— {winner_prob:.1f}% chance</span>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
        rc1, rc2 = st.columns([1, 1], gap="large")
        with rc1:
            st.markdown("<div class='section-header'><i class='fas fa-history'></i><span>Head-to-Head Record</span></div>", unsafe_allow_html=True)
            h2h_rate, h2h_count = get_h2h(home_team, away_team)
            if h2h_count > 0:
                hw = int(h2h_rate * h2h_count); aw = h2h_count - hw
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.markdown(f"<div class='mini-stat'><div class='val' style='color:#4ade80;'>{hw}</div><div class='lbl'>{home_team[:8]} Wins</div></div>", unsafe_allow_html=True)
                with mc2: st.markdown(f"<div class='mini-stat'><div class='val'>{h2h_count}</div><div class='lbl'>Total</div></div>", unsafe_allow_html=True)
                with mc3: st.markdown(f"<div class='mini-stat'><div class='val' style='color:#f87171;'>{aw}</div><div class='lbl'>{away_team[:8]} Wins</div></div>", unsafe_allow_html=True)
            else:
                st.info("No recent head-to-head data found")

        with rc2:
            st.markdown("<div class='section-header'><i class='fas fa-signal'></i><span>Team Comparison</span></div>", unsafe_allow_html=True)
            home_s = team_stats[home_team]; away_s = team_stats[away_team]
            metrics = [
                ("Recent Form", home_s['form'], away_s['form'], True),
                ("Avg Goals Scored", home_s['avg_scored'], away_s['avg_scored'], True),
                ("Avg Goals Conceded", home_s['avg_conceded'], away_s['avg_conceded'], False),
                ("ELO Rating", elo_lookup.get(home_team, 1500), elo_lookup.get(away_team, 1500), True),
            ]
            for label, hv, av, higher_is_better in metrics:
                total = hv + av if (hv + av) > 0 else 1
                h_pct = hv / total * 100; a_pct = av / total * 100
                h_col = '#4ade80' if (hv >= av) == higher_is_better else '#f87171'
                a_col = '#4ade80' if (av > hv) == higher_is_better else '#f87171'
                hv_fmt = f"{hv:.0f}" if hv > 100 else f"{hv:.2f}"
                av_fmt = f"{av:.0f}" if av > 100 else f"{av:.2f}"
                st.markdown(f"""
                <div class='stat-row'>
                    <span style='font-size:0.92rem; font-weight:700; color:{h_col}; width:80px; text-align:right;'>{hv_fmt}</span>
                    <div style='flex:1; text-align:center;'>
                        <div style='font-size:0.75rem; color:#475569; margin-bottom:5px;'>{label}</div>
                        <div style='height:5px; background:#1e293b; border-radius:3px; overflow:hidden; display:flex;'>
                            <div style='width:{h_pct:.0f}%; background:{h_col};'></div>
                            <div style='width:{a_pct:.0f}%; background:{a_col};'></div>
                        </div>
                    </div>
                    <span style='font-size:0.92rem; font-weight:700; color:{a_col}; width:80px;'>{av_fmt}</span>
                </div>""", unsafe_allow_html=True)

# ── PAGE 3: RUN SIMULATION ────────────────────────────────────────────────────
elif page == "Run Simulation":
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; color:#f59e0b;'>
            <i class='fas fa-random' style='margin-right:6px;'></i>Monte Carlo Engine
        </span>
    </div>
    <h1 style='color:#f1f5f9; margin:0 0 6px;'>Live Simulation</h1>
    <p class='hero-sub'>Simulate the full World Cup bracket — group stage through the final</p>
    <hr class='page-divider'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;'>Number of Simulations</p>", unsafe_allow_html=True)
    n_sims = st.slider("", min_value=100, max_value=2000, value=500, step=100, label_visibility="collapsed")

    sc1, sc2, sc3 = st.columns(3)
    with sc1: st.markdown(f"<div class='mini-stat'><div class='val'>{n_sims:,}</div><div class='lbl'>Simulations</div></div>", unsafe_allow_html=True)
    with sc2: st.markdown("<div class='mini-stat'><div class='val'>48</div><div class='lbl'>Teams</div></div>", unsafe_allow_html=True)
    with sc3: st.markdown("<div class='mini-stat'><div class='val'>32</div><div class='lbl'>Features</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Simulation", use_container_width=True)

    if run_btn:
        with st.spinner(f"Simulating {n_sims:,} World Cups..."):
            progress = st.progress(0)
            win_probs = run_simulation(n_sims)
            progress.progress(100)
        st.success(f"✅  {n_sims:,} simulations complete")

        sim_df = pd.DataFrame(list(win_probs.items()), columns=['Team', 'Win Probability (%)'])
        sim_df = sim_df.sort_values('Win Probability (%)', ascending=False).reset_index(drop=True)
        sim_df['Rank'] = sim_df.index + 1

        st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
        champion = sim_df.iloc[0]
        st.markdown(f"""
        <div class='champion-banner'>
            <div class='trophy'>🏆</div>
            <div class='name'>{champion['Team']}</div>
            <div style='font-size:1.2rem; color:#f59e0b; font-weight:700; margin:4px 0;'>{champion['Win Probability (%)']:.1f}%</div>
            <div style='color:#64748b; font-size:0.85rem; margin-top:2px;'>predicted champion in {n_sims:,} simulations</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1], gap="large")
        with rc1:
            st.markdown("<div class='section-header'><i class='fas fa-chart-bar'></i><span>Top 10 — Simulation Results</span></div>", unsafe_allow_html=True)
            top10 = sim_df.head(10).sort_values('Win Probability (%)', ascending=True)
            bar_colors = ['#f59e0b' if t in FAVORITES else '#1e40af' for t in top10['Team']]
            fig, ax = make_dark_chart(figsize=(10, 5))
            bars = ax.barh(top10['Team'], top10['Win Probability (%)'], color=bar_colors, height=0.6)
            for bar, val in zip(bars, top10['Win Probability (%)']):
                ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', ha='left', color='#94a3b8', fontsize=9)
            ax.set_xlabel('Win Probability (%)', color='#64748b', fontsize=10)
            ax.tick_params(axis='y', colors='#cbd5e1', labelsize=10)
            ax.tick_params(axis='x', colors='#475569', labelsize=9)
            ax.set_title(f'Live Simulation Results ({n_sims:,} runs)', color='#94a3b8', fontsize=11, pad=14, loc='left')
            plt.tight_layout(); st.pyplot(fig)
        with rc2:
            st.markdown("<div class='section-header'><i class='fas fa-list-ol'></i><span>Full Rankings</span></div>", unsafe_allow_html=True)
            st.dataframe(sim_df[['Rank', 'Team', 'Win Probability (%)']].head(20), height=420, use_container_width=True, hide_index=True)

# ── PAGE 4: MODEL INFO ────────────────────────────────────────────────────────
elif page == "Model Info":
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <span style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:2.5px; color:#f59e0b;'>
            <i class='fas fa-chart-line' style='margin-right:6px;'></i>XGBoost V3
        </span>
    </div>
    <h1 style='color:#f1f5f9; margin:0 0 6px;'>Model Information</h1>
    <p class='hero-sub'>Architecture, training methodology, and feature importance breakdown</p>
    <hr class='page-divider'>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, val, lbl in [(sc1,"5,831","Training Matches"),(sc2,"56%","Test Accuracy"),(sc3,"32","Features Used"),(sc4,"10,000","MC Simulations")]:
        with col: st.markdown(f"<div class='mini-stat'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><i class='fas fa-code-branch'></i><span>Model Progression</span></div>", unsafe_allow_html=True)

    versions = [
        ("V1", "Baseline", "50.93%", "Rolling form, avg goals, neutral venue flag", "#60a5fa"),
        ("V2", "ELO + Players", "54.83%", "Added ELO ratings and player strength scores", "#a78bfa"),
        ("V3", "Final", "56.00%", "Star player ELO, H2H, squad age, confederation strength, competitive matches only", "#4ade80"),
    ]
    for tag, title, acc, desc, col in versions:
        st.markdown(f"""
        <div class='version-card'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div>
                    <div class='ver-tag'>{tag}</div>
                    <div class='ver-title'>{title}</div>
                    <div class='ver-desc'>{desc}</div>
                </div>
                <div class='ver-acc' style='color:{col};'>{acc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
    fi_c1, fi_c2 = st.columns([2, 1], gap="large")
    with fi_c1:
        st.markdown("<div class='section-header'><i class='fas fa-sort-amount-down'></i><span>Top Features by Importance</span></div>", unsafe_allow_html=True)
        importance_data = {
            'Feature': ['star_elo_diff','elo_diff','h2h','goal_diff','away_age_score','away_mid','defense_diff','attack_diff','form_diff','conf_diff'],
            'Importance': [0.067,0.061,0.043,0.042,0.038,0.034,0.033,0.030,0.030,0.029]
        }
        imp_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
        cmap = LinearSegmentedColormap.from_list('gold_blue', ['#1e40af', '#f59e0b'])
        norm_vals = (imp_df['Importance'] - imp_df['Importance'].min()) / (imp_df['Importance'].max() - imp_df['Importance'].min())
        bar_colors = [cmap(v) for v in norm_vals]
        fig, ax = make_dark_chart(figsize=(10, 6))
        bars = ax.barh(imp_df['Feature'], imp_df['Importance'], color=bar_colors, height=0.6)
        for bar, val in zip(bars, imp_df['Importance']):
            ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', ha='left', color='#94a3b8', fontsize=9)
        ax.set_xlabel('Importance Score', color='#64748b', fontsize=10)
        ax.tick_params(axis='y', colors='#cbd5e1', labelsize=10)
        ax.tick_params(axis='x', colors='#475569', labelsize=9)
        ax.set_title('V3 XGBoost — Feature Importance', color='#94a3b8', fontsize=11, pad=14, loc='left')
        plt.tight_layout(); st.pyplot(fig)

    with fi_c2:
        st.markdown("<div class='section-header'><i class='fas fa-database'></i><span>Data Sources</span></div>", unsafe_allow_html=True)
        sources = [
            ("fa-football-ball","#f59e0b","Match Results","International results 1872–2026 (Kaggle)"),
            ("fa-star","#c084fc","ELO Ratings","WC 2026 ratings for all 48 teams"),
            ("fa-users","#60a5fa","Player Stats","2025-26 season, top 5 European leagues (FBref)"),
            ("fa-user-circle","#4ade80","PlayerElo","Individual player ratings, updated daily"),
        ]
        for icon, color, title, desc in sources:
            st.markdown(f"""
            <div style='display:flex; gap:12px; align-items:flex-start; padding:12px 0; border-bottom:1px solid #1a2235;'>
                <i class='fas {icon}' style='color:{color}; font-size:1.1rem; margin-top:2px; flex-shrink:0;'></i>
                <div>
                    <div style='font-size:0.88rem; font-weight:600; color:#f1f5f9;'>{title}</div>
                    <div style='font-size:0.78rem; color:#64748b; margin-top:2px;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='page-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><i class='fas fa-sitemap'></i><span>Model Pipeline</span></div>", unsafe_allow_html=True)
    steps = [
        ("1","Filter","Competitive matches only (2018–2026) — friendlies removed"),
        ("2","Weight","Recency weights: 2025-26 = 3×, 2023-24 = 2×, 2021-22 = 1.5×, 2018-20 = 1×"),
        ("3","Engineer","32 features per match: ELO, H2H, player strength, age score"),
        ("4","Train","XGBoost with sample weights on 5,831 competitive matches"),
        ("5","Simulate","10,000 Monte Carlo simulations of the full tournament"),
    ]
    pc = st.columns(5)
    for col, (num, title, desc) in zip(pc, steps):
        with col:
            st.markdown(f"""
            <div style='background:#0f1623; border:1px solid #1e293b; border-radius:10px; padding:14px; text-align:center; min-height:140px;'>
                <div style='width:30px; height:30px; border-radius:50%; background:rgba(245,158,11,0.15);
                            border:1px solid #f59e0b; display:flex; align-items:center; justify-content:center;
                            font-size:0.8rem; font-weight:800; color:#f59e0b; margin:0 auto 10px;'>{num}</div>
                <div style='font-size:0.85rem; font-weight:700; color:#f1f5f9; margin-bottom:6px;'>{title}</div>
                <div style='font-size:0.73rem; color:#64748b; line-height:1.4;'>{desc}</div>
            </div>""", unsafe_allow_html=True)