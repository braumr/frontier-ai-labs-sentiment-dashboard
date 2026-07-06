"""Streamlit dashboard for public AI sentiment and risk-signal data."""

import os
from pathlib import Path
from threading import Thread

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_refresh import RefreshConfig, auto_refresh_daily, clean_existing_exports


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

RECORDS_FILE = DATA_DIR / "powerbi_record_level_export.csv"
WEEKLY_FILE = DATA_DIR / "powerbi_weekly_trend_export.csv"
COMPANY_FILE = DATA_DIR / "powerbi_company_topic_summary.csv"
PREDICTION_FILES = [
    DATA_DIR / "powerbi_prediction_results.csv",
    DATA_DIR / "prediction_results_actual_vs_feedforward_nn.csv",
]

BLUE_PALETTE = ["#0B3D91", "#1557B0", "#1F77D0", "#4A90E2", "#7DB5F0", "#A8D0F7"]
LINE_COLORS = {
    "risk": "#0B3D91",
    "general": "#4A90E2",
    "baseline": "#9CA3AF",
    "forecast": "#1F77D0",
}
TRACKED_COMPANIES = {"openai", "anthropic", "google", "microsoft", "meta", "xai"}
NON_COMPANY_VALUES = {"multiple", "general ai", "unknown", "other", "none", ""}


st.set_page_config(
    page_title="Public Sentiment of Frontier AI Labs",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #000000;
            color: #E5E7EB;
        }
        header[data-testid="stHeader"] {
            background: #000000;
            border-bottom: 1px solid #26334D;
        }
        header[data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            color: #E5E7EB;
            background: #000000;
        }
        [data-testid="stToolbar"] {
            background: #000000;
        }
        .block-container {padding-top: 2rem; padding-bottom: 3rem;}
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
        }
        p, li, span, label {
            color: #E5E7EB;
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"],
        .small-note {
            color: #9CA3AF;
        }
        div[data-testid="stMetric"] {
            background: #141A2A;
            border: 1px solid #26334D;
            border-radius: 0.5rem;
            padding: 0.9rem 1rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #9CA3AF;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #E5E7EB;
        }
        .summary-card {
            background: #141A2A;
            border: 1px solid #26334D;
            border-radius: 0.5rem;
            color: #E5E7EB;
            font-size: 1rem;
            line-height: 1.55;
            margin: 1rem 0 1.6rem 0;
            padding: 1rem 1.15rem;
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin: 0.25rem 0 1.6rem 0;
        }
        .insight-card {
            background: #141A2A;
            border: 1px solid #26334D;
            border-left: 3px solid #4A90E2;
            border-radius: 0.5rem;
            padding: 0.9rem 1rem;
        }
        .insight-label {
            color: #9CA3AF;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .insight-value {
            color: #E5E7EB;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .insight-detail {
            color: #9CA3AF;
            font-size: 0.88rem;
            line-height: 1.35;
            margin-top: 0.35rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #26334D;
            border-radius: 0.5rem;
        }
        @media (max-width: 900px) {
            .insight-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 560px) {
            .insight-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path_string: str, modified_time: float) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV without allowing one unavailable or malformed file to stop the app."""
    del modified_time  # Used only to invalidate the cache after a data refresh.
    path = Path(path_string)
    try:
        return pd.read_csv(path), None
    except FileNotFoundError:
        return None, f"{path.name} is not available."
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        return None, f"{path.name} could not be read: {exc}"
    except OSError as exc:
        return None, f"{path.name} could not be opened: {exc}"


def read_optional_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load a CSV and refresh the cache automatically when the file changes."""
    modified_time = path.stat().st_mtime if path.exists() else -1
    return load_csv(str(path), modified_time)


def missing_data_message(message: str | None, purpose: str) -> None:
    del message, purpose
    st.info("Data not available for this section.")


def format_source_name(value: object) -> str:
    source = str(value).replace("_", " ").strip()
    return source.title() if source else "Unknown"


def format_date_range(series: pd.Series) -> tuple[str, str] | None:
    dates = pd.to_datetime(series, errors="coerce", utc=True).dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%b %d, %Y"), dates.max().strftime("%b %d, %Y")


def count_tracked_companies(series: pd.Series) -> int:
    """Count only tracked companies, excluding grouped or generic categories."""
    normalized_values = {
        str(value).strip().lower()
        for value in series.dropna()
        if str(value).strip().lower() not in NON_COMPANY_VALUES
    }
    tracked_matches = normalized_values & TRACKED_COMPANIES
    return len(tracked_matches) if tracked_matches else len(normalized_values)


def filter_tracked_company_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Keep only the main tracked company rows for headline company charts."""
    normalized = df[column].astype(str).str.strip().str.lower()
    return df[normalized.isin(TRACKED_COMPANIES)].copy()


def apply_dark_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the dashboard color system to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font={"color": "#E5E7EB"},
        title_font={"color": "#FFFFFF"},
        legend={"font": {"color": "#E5E7EB"}},
        xaxis={
            "gridcolor": "#26334D",
            "linecolor": "#26334D",
            "tickfont": {"color": "#9CA3AF"},
            "title_font": {"color": "#9CA3AF"},
        },
        yaxis={
            "gridcolor": "#26334D",
            "linecolor": "#26334D",
            "tickfont": {"color": "#9CA3AF"},
            "title_font": {"color": "#9CA3AF"},
        },
    )
    return fig


def get_secret(name: str) -> str | None:
    """Read a credential from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None
    return value or os.getenv(name)


def refresh_config_from_runtime() -> RefreshConfig:
    """Build refresh settings from secrets while keeping original source defaults."""
    return RefreshConfig(
        newsapi_key=get_secret("NEWSAPI_KEY") or get_secret("NEWS_API_KEY"),
        reddit_client_id=get_secret("REDDIT_CLIENT_ID"),
        reddit_client_secret=get_secret("REDDIT_CLIENT_SECRET"),
        reddit_user_agent=get_secret("REDDIT_USER_AGENT")
        or "enterprise-ai-risk-monitor/1.0",
    )


cleanup_result = clean_existing_exports()
if cleanup_result.get("changed"):
    st.cache_data.clear()

if "daily_refresh_started" not in st.session_state:
    Thread(
        target=auto_refresh_daily,
        args=(refresh_config_from_runtime(),),
        daemon=True,
    ).start()
    st.session_state["daily_refresh_started"] = True


records_df, records_error = read_optional_csv(RECORDS_FILE)
weekly_df, weekly_error = read_optional_csv(WEEKLY_FILE)
company_df, company_error = read_optional_csv(COMPANY_FILE)

prediction_df = None
prediction_error = None
prediction_source = None
for prediction_path in PREDICTION_FILES:
    candidate_df, candidate_error = read_optional_csv(prediction_path)
    if candidate_df is not None:
        prediction_df = candidate_df
        prediction_source = prediction_path.name
        break
    prediction_error = candidate_error


st.title("Public Sentiment of Frontier AI Labs")
st.markdown("Let’s explore how the public feels about frontier Artificial Intelligence labs. Data pulled from Reddit and NewsAPI.")
st.markdown(
    """
    <div class="summary-card">
    This dashboard compares public AI sentiment across major frontier AI labs using
    Reddit and NewsAPI records. It highlights which companies receive the strongest
    risk-weighted sentiment, how sentiment changes over time, and how a simple
    neural network forecast compares against a historical baseline.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-label">Highest Risk Sentiment</div>
            <div class="insight-value">OpenAI</div>
            <div class="insight-detail">0.391 avg risk-weighted sentiment</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Most Discussed Company</div>
            <div class="insight-value">OpenAI</div>
            <div class="insight-detail">166 company-specific records</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Risk-Labeled Records</div>
            <div class="insight-value">614</div>
            <div class="insight-detail">Records flagged by risk indicators</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">Dataset Coverage</div>
            <div class="insight-value">Feb 2023 – Jul 2026</div>
            <div class="insight-detail">Reddit + NewsAPI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Which Frontier AI Labs Receive the Strongest Risk Signals?")
st.caption(
    "Risk-weighted sentiment combines sentiment score with keyword-based AI risk "
    "indicators such as misinformation, privacy, cybersecurity, regulation, "
    "copyright, reliability, and job displacement."
)
if company_df is None:
    missing_data_message(company_error, "the company comparison chart")
else:
    category_column = next(
        (
            column
            for column in ["company_or_topic", "company", "topic"]
            if column in company_df.columns
        ),
        None,
    )
    comparison_candidates = {
        "Average Public Risk Sentiment by Company": "avg_risk_sentiment_score",
        "Average Sentiment by Company": "avg_sentiment",
        "Risk-Labeled Record Volume by Company": "risk_records",
        "Total Record Volume by Company": "total_records",
    }
    selected_title = None
    comparison_column = None
    for title, column in comparison_candidates.items():
        if column in company_df.columns:
            selected_title = title
            comparison_column = column
            break

    if category_column and comparison_column:
        company_chart_df = filter_tracked_company_rows(company_df, category_column)
        company_chart_df[comparison_column] = pd.to_numeric(
            company_chart_df[comparison_column], errors="coerce"
        )
        company_chart_df = company_chart_df.dropna(subset=[comparison_column]).sort_values(
            comparison_column, ascending=True
        )
        if company_chart_df.empty:
            st.info("Data not available for this section.")
        else:
            comparison_chart = px.bar(
                company_chart_df,
                x=comparison_column,
                y=category_column,
                orientation="h",
                text_auto=".3f"
                if comparison_column not in {"risk_records", "total_records"}
                else True,
                title="Average Risk-Weighted Sentiment by Company",
                labels={
                    category_column: "Company",
                    comparison_column: "Average risk-weighted sentiment",
                },
                color=category_column,
                color_discrete_sequence=BLUE_PALETTE,
            )
            comparison_chart.update_layout(showlegend=False)
            apply_dark_chart_theme(comparison_chart)
            st.plotly_chart(comparison_chart, width="stretch")
            if comparison_column != "avg_risk_sentiment_score":
                st.caption(f"Chart uses `{comparison_column}`.")
    else:
        st.info("Data not available for this section.")


st.subheader("How Public Sentiment Changes Over Time")
if weekly_df is None:
    missing_data_message(weekly_error, "the weekly trend chart")
elif "week" not in weekly_df.columns:
    st.info("Weekly trend data is not available.")
else:
    weekly_display = weekly_df.copy()
    weekly_display["week"] = pd.to_datetime(weekly_display["week"], errors="coerce")
    weekly_display = weekly_display.dropna(subset=["week"]).sort_values("week")

    weekly_series = []
    if "risk_sentiment_score" in weekly_display.columns:
        weekly_series.append(
            {
                "actual_column": "risk_sentiment_score",
                "rolling_label": "Risk sentiment, 4-week average",
                "rolling_column": "risk_sentiment_score_4wk_avg",
                "color": LINE_COLORS["risk"],
            }
        )
    sentiment_column = next(
        (
            column
            for column in ["avg_sentiment", "average_sentiment", "sentiment"]
            if column in weekly_display.columns
        ),
        None,
    )
    if sentiment_column:
        weekly_series.append(
            {
                "actual_column": sentiment_column,
                "rolling_label": "General sentiment, 4-week average",
                "rolling_column": "avg_sentiment_4wk_avg",
                "color": LINE_COLORS["general"],
            }
        )

    if weekly_series:
        weekly_chart = go.Figure()
        for index, series_config in enumerate(weekly_series):
            actual_column = series_config["actual_column"]
            rolling_column = series_config["rolling_column"]
            weekly_display[actual_column] = pd.to_numeric(
                weekly_display[actual_column], errors="coerce"
            )
            weekly_display[rolling_column] = (
                weekly_display[actual_column].rolling(window=4, min_periods=1).mean()
            )
            color = series_config["color"]
            actual_label = (
                "Risk sentiment, weekly"
                if actual_column == "risk_sentiment_score"
                else "General sentiment, weekly"
            )
            weekly_chart.add_trace(
                go.Scatter(
                    x=weekly_display["week"],
                    y=weekly_display[rolling_column],
                    mode="lines+markers",
                    name=series_config["rolling_label"],
                    visible=True,
                    line={"color": color, "width": 3},
                )
            )
            weekly_chart.add_trace(
                go.Scatter(
                    x=weekly_display["week"],
                    y=weekly_display[actual_column],
                    mode="lines+markers",
                    name=actual_label,
                    visible="legendonly",
                    line={
                        "color": color,
                        "dash": "dash" if actual_column == "risk_sentiment_score" else "dot",
                        "width": 2,
                    },
                )
            )
        weekly_chart.update_layout(
            title="Weekly Public Sentiment Trend",
            xaxis_title="Week",
            yaxis_title="Sentiment score",
            hovermode="x unified",
            legend_title_text="Measure",
        )
        apply_dark_chart_theme(weekly_chart)
        st.plotly_chart(weekly_chart, width="stretch")
        st.caption(
            "4-week rolling averages are shown by default to smooth short-term noise. "
            "Click the legend to show the actual weekly scores."
        )
    else:
        st.info("Weekly trend data is not available.")


st.subheader("Discussion Volume by Company")
if records_df is None:
    missing_data_message(records_error, "the records-by-company chart")
elif "company_or_topic" not in records_df.columns:
    st.info("Data not available for this section.")
else:
    company_counts = (
        filter_tracked_company_rows(records_df, "company_or_topic")["company_or_topic"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("company")
        .reset_index(name="records")
        .sort_values("records", ascending=False)
    )
    if company_counts.empty:
        st.info("Data not available for this section.")
    else:
        records_chart = px.bar(
            company_counts,
            x="records",
            y="company",
            orientation="h",
            text_auto=True,
            title="Collected Records by Company",
            labels={"company": "Company", "records": "Collected records"},
            color="company",
            color_discrete_sequence=BLUE_PALETTE,
        )
        records_chart.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
        apply_dark_chart_theme(records_chart)
        st.plotly_chart(records_chart, width="stretch")
        st.caption("Record volume by company in the collected dataset.")


st.subheader("Next-Week Risk Sentiment Forecast: Neural Network vs Historical Baseline")
if prediction_df is None:
    missing_data_message(prediction_error, "the forecast comparison chart")
else:
    forecast_columns = {
        "Actual next-week score": "actual_next_week_risk_score",
        "Historical mean baseline": "historical_mean_prediction",
        "Feedforward neural network": "feedforward_nn_prediction",
    }
    available_forecasts = {
        label: column
        for label, column in forecast_columns.items()
        if column in prediction_df.columns
    }
    date_column = next(
        (
            column
            for column in ["week", "date", "created_at"]
            if column in prediction_df.columns
        ),
        None,
    )

    if date_column and available_forecasts:
        forecast_display = prediction_df.copy()
        forecast_display[date_column] = pd.to_datetime(
            forecast_display[date_column], errors="coerce"
        )
        forecast_display = forecast_display.dropna(subset=[date_column]).sort_values(
            date_column
        )

        forecast_chart = go.Figure()
        forecast_line_colors = {
            "Actual next-week score": LINE_COLORS["risk"],
            "Historical mean baseline": LINE_COLORS["baseline"],
            "Feedforward neural network": LINE_COLORS["forecast"],
        }
        forecast_line_styles = {
            "Actual next-week score": "solid",
            "Historical mean baseline": "dash",
            "Feedforward neural network": "solid",
        }
        for index, (label, column) in enumerate(available_forecasts.items()):
            forecast_chart.add_trace(
                go.Scatter(
                    x=forecast_display[date_column],
                    y=forecast_display[column],
                    mode="lines+markers",
                    name=label,
                    line={
                        "color": forecast_line_colors.get(label, "#3B82F6"),
                        "dash": forecast_line_styles.get(label, "solid"),
                        "width": 3 if label == "Feedforward neural network" else 2,
                    },
                )
            )
        forecast_chart.update_layout(
            title="Next-Week Risk Sentiment Forecast: Neural Network vs Historical Baseline",
            xaxis_title="Week",
            yaxis_title="Risk-sentiment score",
            hovermode="x unified",
            legend_title_text="Series",
        )
        apply_dark_chart_theme(forecast_chart)
        st.plotly_chart(forecast_chart, width="stretch")
        st.caption(
            "The feedforward neural network was compared against a historical mean "
            "baseline using weekly sentiment features."
        )
    else:
        st.info("Data not available for this section.")


st.subheader("Recent Public Discussion")
if records_df is None:
    missing_data_message(records_error, "the recent-record sample")
else:
    recent_records = records_df.copy()
    record_date_column = next(
        (
            column
            for column in ["created_at", "date", "week"]
            if column in recent_records.columns
        ),
        None,
    )
    if record_date_column:
        recent_records[record_date_column] = pd.to_datetime(
            recent_records[record_date_column], errors="coerce", utc=True
        )
        recent_records = recent_records.sort_values(
            record_date_column, ascending=False, na_position="last"
        )

    column_labels = {
        "created_at": "Date",
        "company_or_topic": "Company",
        "vader_compound": "Sentiment",
        "record_risk_sentiment_score": "Risk Sentiment",
        "risk_label": "Risk Label",
        "title": "Title",
        "url": "Source",
    }
    visible_columns = [column for column in column_labels if column in recent_records.columns]
    if visible_columns:
        display_sample = recent_records[visible_columns].head(5).rename(columns=column_labels)
        column_config = {}
        if "Source" in display_sample.columns:
            column_config["Source"] = st.column_config.LinkColumn(
                "Source", display_text="Open"
            )
        st.dataframe(
            display_sample,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )
    else:
        st.info("Showing the first 5 available rows.")
        st.dataframe(recent_records.head(5), width="stretch", hide_index=True)


st.subheader("Dataset Summary")
if records_df is None:
    missing_data_message(records_error, "record-level KPI metrics")
else:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Collected records", f"{len(records_df):,}")

    company_count = (
        count_tracked_companies(records_df["company_or_topic"])
        if "company_or_topic" in records_df.columns
        else None
    )
    metric_columns[1].metric(
        "Companies",
        f"{company_count:,}" if company_count is not None else "Not available",
    )

    risk_labeled_records = (
        int(pd.to_numeric(records_df["risk_label"], errors="coerce").fillna(0).sum())
        if "risk_label" in records_df.columns
        else None
    )
    metric_columns[2].metric(
        "Risk-labeled records",
        f"{risk_labeled_records:,}"
        if risk_labeled_records is not None
        else "Not available",
    )

    avg_sentiment = (
        pd.to_numeric(records_df["vader_compound"], errors="coerce").mean()
        if "vader_compound" in records_df.columns
        else None
    )
    metric_columns[3].metric(
        "Avg sentiment",
        f"{avg_sentiment:.3f}" if pd.notna(avg_sentiment) else "Not available",
    )

    avg_risk_sentiment = (
        pd.to_numeric(records_df["record_risk_sentiment_score"], errors="coerce").mean()
        if "record_risk_sentiment_score" in records_df.columns
        else None
    )
    metric_columns[4].metric(
        "Avg risk sentiment",
        f"{avg_risk_sentiment:.3f}"
        if pd.notna(avg_risk_sentiment)
        else "Not available",
    )

    date_column = next(
        (column for column in ["created_at", "date", "week"] if column in records_df.columns),
        None,
    )
    date_range = format_date_range(records_df[date_column]) if date_column else None
    if date_range:
        st.caption(f"Available record dates: {date_range[0]} to {date_range[1]}.")


st.subheader("Project Summary")
st.markdown(
    """
    - Collected AI-related public discussion from Reddit and NewsAPI covering OpenAI, Anthropic, Google, Microsoft, Meta, and xAI.
    - Applied VADER sentiment scoring and text-derived features to measure public perception of frontier AI labs.
    - Labeled risk-related discussion using keyword-based indicators for misinformation, privacy, cybersecurity, regulation, reliability, copyright, and job displacement.
    - Aggregated records into weekly company-level sentiment trends and discussion-volume metrics.
    - Trained a feedforward neural network to forecast next-week AI risk sentiment and compared results against a historical mean baseline.
    - Built an interactive Streamlit dashboard to explore company sentiment, risk exposure, public discussion volume, weekly trends, and forecasting results.
    """
)
