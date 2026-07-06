# Public Sentiment of Frontier AI Labs

This project analyzes public sentiment around frontier Artificial Intelligence labs using Reddit and NewsAPI data.

It combines natural language processing, keyword-based risk indicators, weekly trend analysis, and a forecasting experiment to explore public sentiment and risk signals across major frontier AI labs.

## What the project does

- Collects AI-related public discussion from Reddit using PRAW.
- Collects AI-related news articles using NewsAPI.
- Cleans and preprocesses text for analysis.
- Matches records to frontier AI labs or AI-related topics.
- Applies VADER sentiment scoring.
- Labels risk-related discussion using keyword-based risk indicators.
- Uses TF-IDF and other text-derived features in the analysis workflow.
- Aggregates records into weekly company-level sentiment trends.
- Compares a historical mean baseline with a feedforward neural network forecasting experiment.
- Provides a Streamlit dashboard for exploring sentiment, discussion volume, weekly trends, recent records, and forecast results.

## Data sources

- Reddit comments collected with PRAW
- News articles collected with NewsAPI

This duplicate keeps the existing CSV exports in `data/` so the dashboard opens
without API credentials, and it adds an optional refresh workflow for Reddit and
NewsAPI.

## Dashboard

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard reads from the existing CSV files:

- `data/powerbi_record_level_export.csv`
- `data/powerbi_weekly_trend_export.csv`
- `data/powerbi_company_topic_summary.csv`
- `data/powerbi_prediction_results.csv`
- `data/prediction_results_actual_vs_feedforward_nn.csv`

## Streamlit Cloud deployment

Deploy `app.py` from the repository root. Add these values in Streamlit Cloud
Secrets, not in the repository:

```toml
NEWSAPI_KEY = "..."
REDDIT_CLIENT_ID = "..."
REDDIT_CLIENT_SECRET = "..."
REDDIT_USER_AGENT = "..."
```

## Refreshing Reddit and NewsAPI data

The app does not show refresh controls. When the Streamlit app starts or reruns,
it checks `data/refresh_state.json` and automatically refreshes once every 24
hours if credentials are configured.

Set credentials in your shell:

```bash
export NEWSAPI_KEY="..."
export REDDIT_CLIENT_ID="..."
export REDDIT_CLIENT_SECRET="..."
export REDDIT_USER_AGENT="enterprise-ai-risk-monitor/1.0 by your_reddit_username"
```

Or set them in `.streamlit/secrets.toml`. A template is available at:

```text
.streamlit/secrets.toml.example
```

You can also run the refresh from the terminal:

```bash
python data_refresh.py
```

The refresh job uses the original notebook's Reddit subreddits, Reddit search
terms, and company-specific NewsAPI queries. It fetches Reddit comments and
NewsAPI articles, maps them into the existing dashboard schema, appends them to
`data/powerbi_record_level_export.csv`, deduplicates records by URL/title, and
rebuilds:

- `data/powerbi_weekly_trend_export.csv`
- `data/powerbi_company_topic_summary.csv`

The forecasting CSVs are preserved because the original notebook trained that
experiment separately.

## Notebook

The original analysis notebook is:

```text
AI_Risk_Project.ipynb
```

The notebook contains the data collection, preprocessing, sentiment scoring, company/topic matching, risk keyword labeling, weekly feature generation, and forecasting experiment.

## Forecasting experiment

The forecasting portion compares:

- Historical mean baseline
- Feedforward neural network

The model comparison is an experiment using weekly sentiment features. The historical mean baseline is included so the neural network results can be interpreted against a simple reference point.

## Technologies used

- Python
- Pandas
- NumPy
- PRAW
- NewsAPI
- NLTK / VADER sentiment
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Plotly

## Setup

Install the dashboard and refresh dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

To rerun the notebook collection workflow, provide your own Reddit and NewsAPI credentials in your environment or notebook runtime. The notebook workflow may require additional analysis packages such as PRAW, NewsAPI, NLTK, scikit-learn, TensorFlow/Keras, and Jupyter. Do not commit API keys or secrets.
