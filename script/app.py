"""
Weather Dashboard - Streamlit presentation layer.

Reads weather data and ML prediction results from DuckDB only.
No model training, no Open-Meteo calls.

Run:
    streamlit run D:\\weather_etl\\script\\app.py
"""

import time
from datetime import timedelta
from html import escape
import plotly.express as px
import duckdb
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DB_PATH = r"D:\weather_etl\database\weather.duckdb"
REFRESH_SECONDS = 60

COLOR_TEMP = "#38BDF8"
COLOR_HUM = "#22C55E"
COLOR_WIND = "#F59E0B"
COLOR_PRED = "#F43F5E"

BG = "#0B0F17"
CARD_BG = "#111827"
BORDER = "#1F2937"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"

WEATHER_COLS = ["time", "temperature", "humidity", "wind_speed"]
PRED_COLS = [
    "created_at",
    "predicted_temperature",
    "predicted_humidity",
    "predicted_wind_speed",
    "weather_state",
]

RANGE_OPTIONS = {
    "Last 24 Hours": timedelta(hours=24),
    "Last 7 Days": timedelta(days=7),
    "Last 30 Days": timedelta(days=30),
    "All Data": None,
}

STATE_COLORS = {
    "Clear": "#FACC15",
    "Cloudy": "#94A3B8",
    "Rainy": "#38BDF8",
    "Other": "#A78BFA",
}

st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = f"""
<style>
.stApp {{ background-color: {BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{
    background-color: #0E1420;
    border-right: 1px solid {BORDER};
}}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}
.dash-title, .dash-subtitle, .section-title, .card, .card * {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}}
.dash-title {{
    font-size: 2.2rem; font-weight: 700; color: {TEXT};
    margin: 0; letter-spacing: -0.02em;
}}
.dash-subtitle {{ color: {MUTED}; font-size: 1rem; margin-top: 0.25rem; }}
.section-title {{
    font-size: 1.05rem; font-weight: 600; color: {TEXT};
    margin: 2rem 0 0.75rem 0; letter-spacing: 0.01em;
}}
.card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 1.1rem 1.3rem; height: 100%;
}}
.card-label {{
    color: {MUTED}; font-size: 0.78rem; text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 600;
}}
.card-value {{
    font-size: 2.3rem; font-weight: 700; line-height: 1.2;
    margin-top: 0.35rem; word-break: break-word;
}}
.card-unit {{ font-size: 1rem; font-weight: 500; color: {MUTED}; margin-left: 0.25rem; }}
.card-note {{ color: {MUTED}; font-size: 0.8rem; margin-top: 0.4rem; }}
.stat-row {{
    display: flex; justify-content: space-between;
    padding: 0.35rem 0; border-bottom: 1px solid {BORDER};
    font-size: 0.92rem;
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-key {{ color: {MUTED}; }}
.stat-val {{ color: {TEXT}; font-weight: 600; }}
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------------
def run_query(sql: str):
    """Run a read-only query. Returns (DataFrame, error_message_or_None)."""
    last_error = None
    for _ in range(3):
        con = None
        try:
            con = duckdb.connect(DB_PATH, read_only=True)
            df = con.execute(sql).fetchdf()
            return df, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            time.sleep(0.4)  # scheduler may briefly hold the file lock
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:  # noqa: BLE001
                    pass
    return pd.DataFrame(), last_error


def ensure_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Guarantee expected columns exist (filled with NA if missing)."""
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def load_weather_data():
    df, err = run_query("SELECT * FROM weather ORDER BY time ASC")
    if err:
        return pd.DataFrame(columns=WEATHER_COLS + ["time_dt"]), err
    df = ensure_columns(df, WEATHER_COLS)
    df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["temperature", "humidity", "wind_speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time_dt"]).sort_values("time_dt").reset_index(drop=True)
    return df, None


def load_latest_weather():
    df, err = run_query("SELECT * FROM weather ORDER BY time DESC LIMIT 1")
    if err:
        return None, err
    if df.empty:
        return None, None
    return ensure_columns(df, WEATHER_COLS).iloc[0], None


def load_predictions():
    df, err = run_query("SELECT * FROM predictions ORDER BY created_at ASC")
    if err:
        return pd.DataFrame(columns=PRED_COLS), err
    df = ensure_columns(df, PRED_COLS)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    for col in ["predicted_temperature", "predicted_humidity", "predicted_wind_speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, None


def load_latest_prediction():
    df, err = run_query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1")
    if err:
        return None, err
    if df.empty:
        return None, None
    return ensure_columns(df, PRED_COLS).iloc[0], None


def load_metrics_tables():
    """Look for any already-stored metrics tables (never retrains)."""
    tables, err = run_query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE lower(table_name) LIKE '%metric%' "
        "OR lower(table_name) LIKE '%performance%' "
        "OR lower(table_name) LIKE '%evaluation%'"
    )
    if err or tables.empty:
        return {}
    result = {}
    for name in tables["table_name"].tolist():
        safe = str(name).replace('"', '""')
        df, e = run_query(f'SELECT * FROM "{safe}" LIMIT 50')
        if not e and not df.empty:
            result[name] = df
    return result


def filter_weather_data(df: pd.DataFrame, range_label: str) -> pd.DataFrame:
    """Filter relative to the newest weather record in the table."""
    delta = RANGE_OPTIONS.get(range_label)
    if df.empty or delta is None:
        return df
    end = df["time_dt"].max()
    return df[df["time_dt"] >= end - delta]


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def fmt(value, decimals=1, unit=""):
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}{unit}"
    except (TypeError, ValueError):
        return "N/A"


def metric_card(label, value, unit, color, note=""):
    note_html = f'<div class="card-note">{escape(note)}</div>' if note else ""
    return (
        f'<div class="card">'
        f'<div class="card-label">{escape(label)}</div>'
        f'<div class="card-value" style="color:{color};">{escape(str(value))}'
        f'<span class="card-unit">{escape(unit)}</span></div>'
        f"{note_html}</div>"
    )


def section(title):
    st.markdown(
        f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True
    )


def show_error(msg, context):
    st.warning(f"Could not read {context}: {msg}")


def show_table(df: pd.DataFrame):
    """Display a dataframe full-width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Render functions
# ----------------------------------------------------------------------------
def render_header():
    st.markdown('<p class="dash-title">Weather Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dash-subtitle">Live weather ETL + machine learning dashboard '
        "powered by DuckDB, Random Forest regression and classification.</p>",
        unsafe_allow_html=True,
    )


def render_current_weather():
    section("Current Weather")
    latest, err = load_latest_weather()
    if err:
        show_error(err, "the weather table")
        return
    if latest is None:
        st.info("The weather table is empty. Waiting for the ETL to insert data.")
        return

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        metric_card("Temperature", fmt(latest["temperature"]), "°C", COLOR_TEMP),
        unsafe_allow_html=True,
    )
    c2.markdown(
        metric_card("Humidity", fmt(latest["humidity"], 0), "%", COLOR_HUM),
        unsafe_allow_html=True,
    )
    c3.markdown(
        metric_card("Wind Speed", fmt(latest["wind_speed"]), "km/h", COLOR_WIND),
        unsafe_allow_html=True,
    )
    st.caption(f"Latest weather record: {latest['time']}")


def render_weather_state():
    section("Weather State")
    latest, err = load_latest_prediction()
    if err:
        show_error(err, "the predictions table")
        return
    if latest is None:
        st.info("No predictions yet. Waiting for the Scheduler to save one.")
        return

    state = latest["weather_state"]
    state_text = "N/A" if pd.isna(state) else str(state)
    color = STATE_COLORS.get(state_text, COLOR_PRED)

    prob_cols = [
        c for c in latest.index if "prob" in c.lower() or "confidence" in c.lower()
    ]
    note = "Classification probability is not stored in DuckDB."
    if prob_cols:
        parts = []
        for c in prob_cols:
            v = latest[c]
            if pd.notna(v):
                try:
                    v = float(v)
                    parts.append(f"{c}: {v * 100:.1f}%" if v <= 1 else f"{c}: {v:.1f}")
                except (TypeError, ValueError):
                    parts.append(f"{c}: {v}")
        if parts:
            note = " | ".join(parts)

    st.markdown(
        metric_card("Predicted Weather State", state_text, "", color, note),
        unsafe_allow_html=True,
    )


def render_predictions():
    section("ML Prediction")
    latest, err = load_latest_prediction()
    if err:
        show_error(err, "the predictions table")
        return
    if latest is None:
        st.info("No predictions yet. Waiting for the Scheduler to save one.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        metric_card(
            "Predicted Temperature",
            fmt(latest["predicted_temperature"]),
            "°C",
            COLOR_PRED,
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        metric_card(
            "Predicted Humidity", fmt(latest["predicted_humidity"]), "%", COLOR_PRED
        ),
        unsafe_allow_html=True,
    )
    c3.markdown(
        metric_card(
            "Predicted Wind Speed",
            fmt(latest["predicted_wind_speed"]),
            "km/h",
            COLOR_PRED,
        ),
        unsafe_allow_html=True,
    )
    state = "N/A" if pd.isna(latest["weather_state"]) else str(latest["weather_state"])
    c4.markdown(
        metric_card("Weather State", state, "", STATE_COLORS.get(state, COLOR_PRED)),
        unsafe_allow_html=True,
    )
    st.caption(f"Prediction created at: {latest['created_at']}")


def render_charts(filtered: pd.DataFrame, range_label: str):
    section(f"Historical Weather ({range_label})")

    if filtered.empty:
        st.info("No weather data available for the selected time range.")
        return

    series = [
        ("Temperature (°C)", "temperature", COLOR_TEMP),
        ("Humidity (%)", "humidity", COLOR_HUM),
        ("Wind Speed (km/h)", "wind_speed", COLOR_WIND),
    ]

    for title, col, color in series:
        st.markdown(f"**{title}**")

        plot_df = filtered[["time_dt", col]].dropna().sort_values("time_dt")

        if plot_df.empty:
            st.info(f"No {col} data available.")
            continue

        fig = px.line(
            plot_df,
            x="time_dt",
            y=col,
            markers=True,
        )

        fig.update_traces(
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            connectgaps=False,
            hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
        )

        fig.update_layout(
            height=320,
            template="plotly_dark",
            paper_bgcolor=BG,
            plot_bgcolor=CARD_BG,
            showlegend=False,
            xaxis_title="Time",
            yaxis_title=title,
            margin=dict(l=10, r=10, t=20, b=10),
        )

        fig.update_yaxes(autorange=True, showgrid=True, gridcolor=BORDER)
        fig.update_xaxes(showgrid=True, gridcolor=BORDER)

        st.plotly_chart(fig, use_container_width=True)


def render_statistics(filtered: pd.DataFrame, range_label: str):
    section(f"Statistics ({range_label})")
    if filtered.empty:
        st.info("No statistics available for the selected time range.")
        return

    items = [
        ("Temperature", "temperature", "°C", COLOR_TEMP, 1),
        ("Humidity", "humidity", "%", COLOR_HUM, 1),
        ("Wind Speed", "wind_speed", "km/h", COLOR_WIND, 1),
    ]
    cols = st.columns(3)
    for col_ui, (label, col, unit, color, dec) in zip(cols, items):
        s = filtered[col]
        rows = [("Average", s.mean()), ("Minimum", s.min()), ("Maximum", s.max())]
        body = "".join(
            f'<div class="stat-row"><span class="stat-key">{k}</span>'
            f'<span class="stat-val">{fmt(v, dec)} {unit}</span></div>'
            for k, v in rows
        )
        col_ui.markdown(
            f'<div class="card"><div class="card-label" style="color:{color};">{label}</div>'
            f'<div style="margin-top:0.6rem;">{body}</div></div>',
            unsafe_allow_html=True,
        )


def render_ml_performance():
    section("ML Performance")
    metrics = load_metrics_tables()
    if not metrics:
        st.info(
            "Regression performance metrics are not stored in DuckDB, "
            "so none are shown. The dashboard does not retrain models."
        )
        return
    for name, df in metrics.items():
        st.markdown(f"**{name}**")
        show_table(df)


def render_prediction_history():
    section("Prediction History")
    df, err = load_predictions()
    if err:
        show_error(err, "the predictions table")
        return
    if df.empty:
        st.info("No prediction history yet.")
        return

    chart_df = df.dropna(subset=["created_at"]).set_index("created_at")[
        ["predicted_temperature", "predicted_humidity", "predicted_wind_speed"]
    ]
    if len(chart_df) >= 2:
        st.line_chart(
            chart_df,
            color=[COLOR_TEMP, COLOR_HUM, COLOR_WIND],
            height=300,
        )
    else:
        st.caption("At least two predictions are needed to draw a history chart.")

    table = df.sort_values("created_at", ascending=False).reset_index(drop=True)
    show_table(table)


def render_dataset_info():
    section("Dataset Information")
    w, w_err = run_query(
        "SELECT COUNT(*) AS n, MIN(time) AS first_t, MAX(time) AS last_t FROM weather"
    )
    p, p_err = run_query("SELECT COUNT(*) AS n FROM predictions")

    weather_n = int(w["n"].iloc[0]) if not w_err and not w.empty else None
    first_t = w["first_t"].iloc[0] if not w_err and not w.empty else None
    last_t = w["last_t"].iloc[0] if not w_err and not w.empty else None
    pred_n = int(p["n"].iloc[0]) if not p_err and not p.empty else None

    if w_err:
        show_error(w_err, "the weather table")
    if p_err:
        show_error(p_err, "the predictions table")

    def txt(v):
        return "N/A" if v is None or pd.isna(v) else str(v)

    def count(v):
        return "N/A" if v is None else f"{v:,}"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        metric_card("Weather Records", count(weather_n), "", TEXT),
        unsafe_allow_html=True,
    )
    c2.markdown(
        metric_card("Prediction Records", count(pred_n), "", TEXT),
        unsafe_allow_html=True,
    )
    c3.markdown(
        metric_card("Earliest Timestamp", txt(first_t), "", TEXT),
        unsafe_allow_html=True,
    )
    c4.markdown(
        metric_card("Latest Timestamp", txt(last_t), "", TEXT), unsafe_allow_html=True
    )


# ----------------------------------------------------------------------------
# Dashboard body (auto-refreshing)
# ----------------------------------------------------------------------------
def dashboard_body(range_label: str):
    weather_df, w_err = load_weather_data()
    if w_err:
        show_error(w_err, "the weather table")
    filtered = filter_weather_data(weather_df, range_label)

    render_current_weather()
    render_weather_state()
    render_predictions()
    render_charts(filtered, range_label)
    render_statistics(filtered, range_label)
    render_ml_performance()
    render_prediction_history()
    render_dataset_info()

    st.caption(f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


HAS_FRAGMENT = hasattr(st, "fragment")

if HAS_FRAGMENT:

    @st.fragment(run_every=REFRESH_SECONDS)
    def auto_refreshing_dashboard(range_label: str):
        dashboard_body(range_label)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    with st.sidebar:
        st.markdown("### Filters")
        range_label = st.selectbox("Time Range", list(RANGE_OPTIONS.keys()), index=1)
        st.caption("The time range affects historical charts and statistics only.")
        st.markdown("---")
        st.caption(f"Auto-refresh every {REFRESH_SECONDS} seconds")
        st.caption(f"Database: {DB_PATH}")

    render_header()

    if HAS_FRAGMENT:
        auto_refreshing_dashboard(range_label)
    else:
        # Fallback for older Streamlit versions without st.fragment
        dashboard_body(range_label)
        time.sleep(REFRESH_SECONDS)
        st.rerun()


main()
