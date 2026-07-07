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

BLUE_PALETTE = ["#7ec8f8", "#67b7eb", "#54a6dd", "#4597cf", "#3586bd", "#2a75a7"]
LINE_COLORS = {
    "risk": "#7ec8f8",
    "general": "#b7ddff",
    "baseline": "#9CA3AF",
    "forecast": "#7ec8f8",
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
        :root {
            --main-bg: #0f131a;
            --sidebar-bg: #2a2d35;
            --control-bg: #11161d;
            --table-header: #1f222a;
            --active-sidebar: #4b4e58;
            --text: #f4f4f5;
            --muted: #b9bec8;
            --accent: #7ec8f8;
            --grid: rgba(185, 190, 200, 0.16);
        }
        .stApp {
            background: var(--main-bg);
            color: var(--text);
        }
        header[data-testid="stHeader"] {
            background: var(--main-bg);
            border-bottom: none;
        }
        header[data-testid="stHeader"] button,
        [data-testid="stToolbar"] button,
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
            color: var(--text);
            background: var(--main-bg);
        }
        [data-testid="stToolbar"] {
            background: var(--main-bg);
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
        }
        [data-testid="stSidebar"] > div {
            background: var(--sidebar-bg);
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text) !important;
            font-weight: 650;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
            border-radius: 0.4rem;
            margin: 0.2rem 0;
            padding: 0.28rem 0.55rem;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--active-sidebar);
        }
        .block-container {
            max-width: 1520px;
            padding-top: 4.8rem;
            padding-bottom: 4rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        h1, h2, h3, h4, h5, h6,
        h1 *, h2 *, h3 *, h4 *, h5 *, h6 * {
            color: #FFFFFF !important;
            letter-spacing: 0;
        }
        h1 {
            font-size: 3rem !important;
            font-weight: 800 !important;
            line-height: 1.08 !important;
            margin-bottom: 1rem !important;
        }
        h2, h3 {
            font-weight: 750 !important;
            margin-top: 1.6rem !important;
        }
        p, li, span, label {
            color: var(--text);
        }
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"],
        .small-note {
            color: var(--muted);
        }
        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.2rem 0;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 750;
        }
        .hero-wrap {
            align-items: center;
            display: grid;
            gap: 3rem;
            grid-template-columns: minmax(0, 0.9fr) minmax(360px, 1.2fr);
            margin-top: 1.5rem;
            min-height: 460px;
        }
        .hero-subtitle {
            color: var(--text);
            font-size: 1.08rem;
            font-weight: 650;
            line-height: 1.65;
            margin: 1.2rem 0 1.8rem 0;
            max-width: 660px;
        }
        .coverage-line {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 650;
        }
        .image-placeholder {
            align-items: center;
            aspect-ratio: 16 / 9;
            background:
                radial-gradient(circle at 50% 40%, rgba(126, 200, 248, 0.15), transparent 28%),
                linear-gradient(145deg, #05070a 0%, #0b1017 55%, #07090d 100%);
            border: 1px solid rgba(185, 190, 200, 0.12);
            border-radius: 0.55rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
            color: var(--muted);
            display: flex;
            justify-content: center;
            min-height: 300px;
            overflow: hidden;
            width: 100%;
        }
        .image-placeholder span {
            color: var(--muted);
            font-size: 0.95rem;
            letter-spacing: 0;
        }
        .thin-divider {
            border-top: 1px solid rgba(185, 190, 200, 0.24);
            margin: 2.2rem 0 2rem 0;
        }
        .plain-two-col {
            display: grid;
            gap: 4rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .plain-two-col h3 {
            font-size: 1.45rem !important;
            margin-bottom: 1rem !important;
        }
        .plain-two-col p {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.7;
        }
        .plain-two-col .muted {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 1.35rem;
        }
        .section-spacer {
            height: 1.15rem;
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stMultiSelect"] > div,
        div[data-testid="stTextInput"] > div {
            background: var(--control-bg);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(185, 190, 200, 0.14);
            border-radius: 0.35rem;
            overflow: hidden;
        }
        div[data-testid="stDataFrame"] [role="columnheader"] {
            background: var(--table-header) !important;
            color: var(--text) !important;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
            .hero-wrap,
            .plain-two-col {
                grid-template-columns: 1fr;
            }
            h1 {
                font-size: 2.2rem !important;
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
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#f4f4f5"},
        title_font={"color": "#FFFFFF"},
        legend={"font": {"color": "#f4f4f5"}},
        xaxis={
            "gridcolor": "rgba(185, 190, 200, 0.16)",
            "linecolor": "rgba(185, 190, 200, 0.22)",
            "tickfont": {"color": "#b9bec8"},
            "title_font": {"color": "#b9bec8"},
        },
        yaxis={
            "gridcolor": "rgba(185, 190, 200, 0.16)",
            "linecolor": "rgba(185, 190, 200, 0.22)",
            "tickfont": {"color": "#b9bec8"},
            "title_font": {"color": "#b9bec8"},
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


def get_dataset_summary(df: pd.DataFrame | None) -> dict[str, str | None]:
    """Build display-ready top-level metrics from the current record export."""
    if df is None:
        return {
            "Collected records": None,
            "Companies": None,
            "Risk-labeled records": None,
            "Avg sentiment": None,
            "Avg risk sentiment": None,
            "Coverage": None,
        }

    company_count = (
        count_tracked_companies(df["company_or_topic"])
        if "company_or_topic" in df.columns
        else None
    )
    risk_labeled_records = (
        int(pd.to_numeric(df["risk_label"], errors="coerce").fillna(0).sum())
        if "risk_label" in df.columns
        else None
    )
    avg_sentiment = (
        pd.to_numeric(df["vader_compound"], errors="coerce").mean()
        if "vader_compound" in df.columns
        else None
    )
    avg_risk_sentiment = (
        pd.to_numeric(df["record_risk_sentiment_score"], errors="coerce").mean()
        if "record_risk_sentiment_score" in df.columns
        else None
    )
    date_column = next(
        (column for column in ["created_at", "date", "week"] if column in df.columns),
        None,
    )
    date_range = format_date_range(df[date_column]) if date_column else None

    return {
        "Collected records": f"{len(df):,}",
        "Companies": f"{company_count:,}" if company_count is not None else "Not available",
        "Risk-labeled records": (
            f"{risk_labeled_records:,}"
            if risk_labeled_records is not None
            else "Not available"
        ),
        "Avg sentiment": (
            f"{avg_sentiment:.3f}" if pd.notna(avg_sentiment) else "Not available"
        ),
        "Avg risk sentiment": (
            f"{avg_risk_sentiment:.3f}"
            if pd.notna(avg_risk_sentiment)
            else "Not available"
        ),
        "Coverage": f"{date_range[0]} to {date_range[1]}" if date_range else None,
    }


def render_metric_row(summary: dict[str, str | None]) -> None:
    metric_columns = st.columns(5)
    for column, label in zip(
        metric_columns,
        [
            "Collected records",
            "Companies",
            "Risk-labeled records",
            "Avg sentiment",
            "Avg risk sentiment",
        ],
    ):
        column.metric(label, summary.get(label) or "Not available")


def render_company_risk_chart() -> None:
    st.subheader("Average Risk-Weighted Sentiment by Company")
    if company_df is None:
        missing_data_message(company_error, "the company comparison chart")
        return

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
    comparison_column = None
    for column in comparison_candidates.values():
        if column in company_df.columns:
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
                color_discrete_sequence=[BLUE_PALETTE[0]],
            )
            comparison_chart.update_layout(showlegend=False)
            apply_dark_chart_theme(comparison_chart)
            st.plotly_chart(comparison_chart, width="stretch")
            if comparison_column != "avg_risk_sentiment_score":
                st.caption(f"Chart uses `{comparison_column}`.")
    else:
        st.info("Data not available for this section.")


def render_weekly_sentiment_chart(
    source_df: pd.DataFrame | None = None,
    error: str | None = None,
    title: str = "Weekly Public Sentiment Trend",
) -> None:
    st.subheader(title)
    if source_df is None:
        missing_data_message(error, "the weekly trend chart")
        return
    if "week" not in source_df.columns:
        st.info("Weekly trend data is not available.")
        return

    weekly_display = source_df.copy()
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
            title=title,
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


def render_discussion_volume_chart(source_df: pd.DataFrame | None = None) -> None:
    st.subheader("Discussion Volume by Company")
    if source_df is None:
        missing_data_message(records_error, "the records-by-company chart")
        return
    if "company_or_topic" not in source_df.columns:
        st.info("Data not available for this section.")
        return

    company_counts = (
        filter_tracked_company_rows(source_df, "company_or_topic")["company_or_topic"]
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
            color_discrete_sequence=[BLUE_PALETTE[0]],
        )
        records_chart.update_layout(showlegend=False, yaxis={"autorange": "reversed"})
        apply_dark_chart_theme(records_chart)
        st.plotly_chart(records_chart, width="stretch")
        st.caption("Record volume by company in the collected dataset.")


def render_forecast_chart() -> None:
    st.subheader("Next-Week Risk Sentiment Forecast: Neural Network vs Historical Baseline")
    if prediction_df is None:
        missing_data_message(prediction_error, "the forecast comparison chart")
        return

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


def prepare_recent_records(source_df: pd.DataFrame) -> pd.DataFrame:
    recent_records = source_df.copy()
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
    return recent_records


def render_records_table(source_df: pd.DataFrame | None = None, limit: int | None = 5) -> None:
    st.subheader("Recent Public Discussion")
    if source_df is None:
        missing_data_message(records_error, "the recent-record sample")
        return

    recent_records = prepare_recent_records(source_df)

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
        display_sample = recent_records[visible_columns]
        if limit is not None:
            display_sample = display_sample.head(limit)
        display_sample = display_sample.rename(columns=column_labels)
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
        if limit is not None:
            st.info(f"Showing the first {limit} available rows.")
            recent_records = recent_records.head(limit)
        st.dataframe(recent_records, width="stretch", hide_index=True)


def render_dataset_summary(source_df: pd.DataFrame | None = None) -> None:
    st.subheader("Dataset Summary")
    if source_df is None:
        missing_data_message(records_error, "record-level KPI metrics")
        return
    summary = get_dataset_summary(source_df)
    render_metric_row(summary)
    if summary.get("Coverage"):
        st.caption(f"Available record dates: {summary['Coverage']}.")


def render_landing_page() -> None:
    summary = get_dataset_summary(records_df)
    coverage = summary.get("Coverage") or "Coverage not available"
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div>
                <h1>Public Sentiment of Frontier AI Labs</h1>
                <div class="hero-subtitle">
                    A machine learning dashboard for analyzing public AI risk sentiment
                    across major frontier AI labs using Reddit and NewsAPI records.
                </div>
                <div class="coverage-line">Dataset coverage: {coverage} · Reddit + NewsAPI</div>
            </div>
            <div class="image-placeholder">
                <span>AI-themed image placeholder</span>
            </div>
        </div>
        <div class="thin-divider"></div>
        <div class="plain-two-col">
            <div>
                <h3>What It Does</h3>
                <p>
                    Collects AI-related public discussion, scores sentiment, labels
                    risk-related records, and summarizes how public perception differs
                    across frontier AI labs.
                </p>
                <p class="muted">
                    The dashboard focuses on public discussion signals, not objective
                    measurements of company safety or technical risk.
                </p>
            </div>
            <div>
                <h3>Method</h3>
                <p>
                    Uses VADER sentiment, keyword-based AI risk indicators, weekly
                    aggregation, and a feedforward neural network forecast compared
                    against a historical mean baseline.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_dashboard() -> None:
    st.title("Overview Dashboard")
    if records_df is None:
        missing_data_message(records_error, "record-level KPI metrics")
    else:
        render_metric_row(get_dataset_summary(records_df))
        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    render_company_risk_chart()
    render_discussion_volume_chart(records_df)
    render_weekly_sentiment_chart(weekly_df, weekly_error, "Weekly Public Sentiment Trend")
    render_dataset_summary(records_df)


def render_company_explorer() -> None:
    st.title("Company Explorer")
    if records_df is None or "company_or_topic" not in records_df.columns:
        missing_data_message(records_error, "company-level records")
        return

    company_options = sorted(
        value
        for value in records_df["company_or_topic"].dropna().astype(str).unique()
        if value.strip().lower() in TRACKED_COMPANIES
    )
    if not company_options:
        st.info("Company-specific records are not available.")
        return

    selected_company = st.selectbox("Company", company_options)
    company_records = records_df[
        records_df["company_or_topic"].astype(str).str.lower() == selected_company.lower()
    ].copy()
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    render_metric_row(get_dataset_summary(company_records))

    left, right = st.columns(2)
    if "risk_label" in company_records.columns:
        risk_counts = (
            pd.to_numeric(company_records["risk_label"], errors="coerce")
            .fillna(0)
            .astype(int)
            .map({0: "Not risk-labeled", 1: "Risk-labeled"})
            .value_counts()
            .rename_axis("label")
            .reset_index(name="records")
        )
        risk_chart = px.bar(
            risk_counts,
            x="label",
            y="records",
            title=f"{selected_company} Risk-Labeled Records",
            labels={"label": "", "records": "Records"},
            color_discrete_sequence=[BLUE_PALETTE[0]],
        )
        apply_dark_chart_theme(risk_chart)
        left.plotly_chart(risk_chart, width="stretch")

    if "source" in company_records.columns:
        source_counts = (
            company_records["source"]
            .fillna("Unknown")
            .map(format_source_name)
            .value_counts()
            .rename_axis("source")
            .reset_index(name="records")
        )
        source_chart = px.bar(
            source_counts,
            x="source",
            y="records",
            title=f"{selected_company} Records by Source",
            labels={"source": "", "records": "Records"},
            color_discrete_sequence=[BLUE_PALETTE[0]],
        )
        apply_dark_chart_theme(source_chart)
        right.plotly_chart(source_chart, width="stretch")

    if {"week", "vader_compound", "record_risk_sentiment_score"}.issubset(
        company_records.columns
    ):
        weekly_company = company_records.copy()
        weekly_company["week"] = pd.to_datetime(weekly_company["week"], errors="coerce")
        weekly_company = weekly_company.dropna(subset=["week"])
        weekly_company = (
            weekly_company.groupby("week", as_index=False)
            .agg(
                total_records=("week", "size"),
                avg_sentiment=("vader_compound", "mean"),
                risk_sentiment_score=("record_risk_sentiment_score", "mean"),
            )
            .sort_values("week")
        )
        render_weekly_sentiment_chart(
            weekly_company,
            None,
            f"{selected_company} Weekly Public Sentiment Trend",
        )

    render_records_table(company_records, limit=8)


def render_forecast_explorer() -> None:
    st.title("Forecast Explorer")
    render_forecast_chart()
    st.caption(
        "The forecast compares the feedforward neural network against a historical "
        "mean baseline using weekly sentiment features."
    )


def render_records_explorer() -> None:
    st.title("Records Explorer")
    if records_df is None:
        missing_data_message(records_error, "the recent-record sample")
        return

    filtered_records = records_df.copy()
    filter_columns = st.columns(3)
    if "company_or_topic" in filtered_records.columns:
        company_options = sorted(
            filtered_records["company_or_topic"].dropna().astype(str).unique()
        )
        selected_companies = filter_columns[0].multiselect(
            "Company", company_options, default=[]
        )
        if selected_companies:
            filtered_records = filtered_records[
                filtered_records["company_or_topic"].astype(str).isin(selected_companies)
            ]
    if "source" in filtered_records.columns:
        source_options = sorted(filtered_records["source"].dropna().astype(str).unique())
        selected_sources = filter_columns[1].multiselect("Source", source_options, default=[])
        if selected_sources:
            filtered_records = filtered_records[
                filtered_records["source"].astype(str).isin(selected_sources)
            ]
    if "risk_label" in filtered_records.columns:
        risk_filter = filter_columns[2].selectbox(
            "Risk label", ["All", "Risk-labeled", "Not risk-labeled"]
        )
        risk_values = pd.to_numeric(filtered_records["risk_label"], errors="coerce").fillna(0)
        if risk_filter == "Risk-labeled":
            filtered_records = filtered_records[risk_values == 1]
        elif risk_filter == "Not risk-labeled":
            filtered_records = filtered_records[risk_values == 0]

    render_records_table(filtered_records, limit=None)


pages = [
    "app",
    "Overview Dashboard",
    "Company Explorer",
    "Forecast Explorer",
    "Records Explorer",
]
page = st.sidebar.radio("Navigation", pages, label_visibility="collapsed")

if page == "app":
    render_landing_page()
elif page == "Overview Dashboard":
    render_overview_dashboard()
elif page == "Company Explorer":
    render_company_explorer()
elif page == "Forecast Explorer":
    render_forecast_explorer()
elif page == "Records Explorer":
    render_records_explorer()
