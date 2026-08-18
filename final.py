import io
import re
from collections import deque
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set Streamlit page layout
st.set_page_config(
    page_title="Global Telecom Analytics, Stack & Queue Engine",
    layout="wide",
    page_icon="🌍",
)


# ==========================================
# 1. ALGORITHMIC COUNTRY CLEANER
# ==========================================
def dynamic_clean_country(name):
    """Standardizes country names dynamically without dictionary overhead."""
    if pd.isna(name):
        return np.nan

    cleaned = re.sub(r"\.", "", str(name).strip()).title()

    acronym_map = {
        "Usa": "United States",
        "Uk": "United Kingdom",
        "Prc": "China",
        "Ind": "India",
        "Rsa": "South Africa",
        "Esp": "Spain",
        "Fra": "France",
        "Ger": "Germany",
        "Bra": "Brazil",
        "Jpn": "Japan",
        "Can": "Canada",
        "Aus": "Australia",
        "Mex": "Mexico",
        "Ita": "Italy",
        "Rok": "South Korea",
        "Rus": "Russia",
        "Nga": "Nigeria",
        "Idn": "Indonesia",
    }
    return acronym_map.get(cleaned, cleaned)


# ==========================================
# 2. QUEUE & STACK IMPLEMENTATIONS
# ==========================================
def compute_rolling_avg_deque(series, window=3):
    """Queue Pattern: Computes a rolling average manually using collections.deque(maxlen=3).

    Each new year's value is pushed into the deque. Once length exceeds 3,
    the deque automatically evicts the oldest year (FIFO).
    """
    d = deque(maxlen=window)
    result = []
    for val in series:
        if pd.isna(val):
            result.append(np.nan)
        else:
            d.append(val)
            result.append(sum(d) / len(d))
    return pd.Series(result, index=series.index)


def detect_stack_anomalies(df_cleaned):
    """Monotonic Stack Pattern: Audits country adoption series for chronological drops.

    Pushes years onto a monotonic stack as values rise/hold steady.
    When a drop occurs (Value < Top of Stack), pops the stack and flags an anomaly.
    """
    anomalies = []
    df_valid = df_cleaned.dropna(subset=["internet_users_pct"]).sort_values(
        ["Country", "Year"]
    )

    for country, group in df_valid.groupby("Country"):
        stack = []  # Stores tuples: (year, value)

        for _, row in group.iterrows():
            year = int(row["Year"])
            val = float(row["internet_users_pct"])

            # If current value drops below the top of stack
            if stack and val < stack[-1][1]:
                prev_year, prev_val = stack[-1]
                anomalies.append({
                    "Country": country,
                    "Year": year,
                    "Drop Value (%)": val,
                    "Previous Peak Year": prev_year,
                    "Previous Peak (%)": prev_val,
                    "Drop Magnitude (% Pts)": prev_val - val,
                })

                # Maintain monotonic property by popping peaks greater than current value
                while stack and stack[-1][1] > val:
                    stack.pop()

            stack.append((year, val))

    df_anomalies = pd.DataFrame(anomalies)
    if not df_anomalies.empty:
        df_anomalies = df_anomalies.sort_values(
            "Drop Magnitude (% Pts)", ascending=False
        ).reset_index(drop=True)
    return df_anomalies


# ==========================================
# 3. PIPELINE: MERGE, CLEAN, QUEUE & GROWTH
# ==========================================
def process_and_clean_telecom_data(
    df_internet_raw, df_mobile_raw, missing_strategy="impute"
):
    logs = {}

    id_vars = ["Country", "Country Code", "Indicator Name", "Indicator Code"]
    year_cols = [
        c for c in df_internet_raw.columns if c not in id_vars and str(c).strip()
    ]

    df_int_long = pd.melt(
        df_internet_raw,
        id_vars=[c for c in id_vars if c in df_internet_raw.columns],
        value_vars=[c for c in year_cols if c in df_internet_raw.columns],
        var_name="Year",
        value_name="internet_users_pct",
    )

    df_mob_long = pd.melt(
        df_mobile_raw,
        id_vars=[c for c in id_vars if c in df_mobile_raw.columns],
        value_vars=[c for c in year_cols if c in df_mobile_raw.columns],
        var_name="Year",
        value_name="mobile_subscriptions",
    )

    # Merge
    df_merged = pd.merge(
        df_int_long,
        df_mob_long,
        on=["Country", "Year"],
        how="outer",
        suffixes=("_int", "_mob"),
    )
    logs["raw_rows"] = len(df_merged)

    # Clean country names & deduplicate
    df_merged = df_merged.drop_duplicates().reset_index(drop=True)
    df_merged["Country"] = df_merged["Country"].apply(dynamic_clean_country)
    df_merged = df_merged.drop_duplicates().reset_index(drop=True)

    # Convert Types
    df_merged["Year"] = pd.to_numeric(
        df_merged["Year"], errors="coerce"
    ).astype("Int64")

    if "internet_users_pct" in df_merged.columns:
        df_merged["internet_users_pct"] = (
            df_merged["internet_users_pct"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df_merged["internet_users_pct"] = pd.to_numeric(
            df_merged["internet_users_pct"], errors="coerce"
        )

    if "mobile_subscriptions" in df_merged.columns:
        df_merged["mobile_subscriptions"] = pd.to_numeric(
            df_merged["mobile_subscriptions"], errors="coerce"
        )

    # Outlier Flagging and Correction
    pct_outliers = (df_merged["internet_users_pct"] < 0) | (
        df_merged["internet_users_pct"] > 100
    )
    logs["pct_outliers_flagged"] = int(pct_outliers.sum())
    df_merged.loc[pct_outliers, "internet_users_pct"] = np.nan

    sub_outliers = df_merged["mobile_subscriptions"] < 0
    logs["sub_outliers_corrected"] = int(sub_outliers.sum())
    df_merged["mobile_subscriptions"] = df_merged["mobile_subscriptions"].apply(
        lambda x: abs(x) if pd.notna(x) and x < 0 else x
    )

    # Missing Value Handling Strategy
    if missing_strategy == "impute":
        df_merged = df_merged.sort_values(by=["Country", "Year"]).reset_index(
            drop=True
        )
        for col in ["internet_users_pct", "mobile_subscriptions"]:
            df_merged[col] = df_merged.groupby("Country")[col].transform(
                lambda grp: grp.ffill().bfill()
            )
    elif missing_strategy == "drop":
        df_merged = df_merged.dropna(
            subset=["internet_users_pct", "mobile_subscriptions"]
        ).reset_index(drop=True)

    # Manual Queue (deque) 3-Year Rolling Average Computation
    df_merged = df_merged.sort_values(["Country", "Year"]).reset_index(
        drop=True
    )
    df_merged["internet_pct_3yr_avg"] = df_merged.groupby(
        "Country", group_keys=False
    ).apply(
        lambda grp: compute_rolling_avg_deque(
            grp["internet_users_pct"], window=3
        )
    )

    logs["final_rows"] = len(df_merged)
    return df_merged, logs


def compute_country_growth(df_cleaned):
    """Computes total percentage points growth (last_val - first_val) per country."""
    df_valid = df_cleaned.dropna(subset=["internet_users_pct"]).sort_values(
        ["Country", "Year"]
    )

    growth_df = (
        df_valid.groupby("Country")
        .agg(
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            first_pct=("internet_users_pct", "first"),
            last_pct=("internet_users_pct", "last"),
        )
        .reset_index()
    )

    growth_df["pct_points_growth"] = (
        growth_df["last_pct"] - growth_df["first_pct"]
    )
    growth_df["duration_years"] = (
        growth_df["last_year"] - growth_df["first_year"]
    )
    return growth_df.sort_values("pct_points_growth", ascending=False)


# ==========================================
# 4. STREAMLIT APPLICATION & UI
# ==========================================
st.title("🌍 Global Telecom Analytics & Data Structure Engine")

# Sidebar
st.sidebar.header("⚙️ Data Pipeline Configuration")
missing_strategy = st.sidebar.radio(
    "Missing Value Handling Strategy:",
    options=["impute", "drop"],
    format_func=lambda x: (
        "Impute (Time Series ffill/bfill)" if x == "impute" else "Drop Rows"
    ),
)

st.sidebar.subheader("📁 Upload Custom Files")
file_internet = st.sidebar.file_uploader("Upload Internet CSV", type=["csv"])
file_mobile = st.sidebar.file_uploader("Upload Mobile CSV", type=["csv"])

# Data Loader
if file_internet and file_mobile:
    df_int = pd.read_csv(file_internet)
    df_mob = pd.read_csv(file_mobile)
else:
    try:
        df_int = pd.read_csv("C:\\Users\\Dell\\Desktop\\SEMMANAGER\\SIC\\Repository1\\internet.csv")
        df_mob = pd.read_csv("C:\\Users\\Dell\\Desktop\\SEMMANAGER\\SIC\\Repository1\\mobile.csv")
    except Exception:
        st.warning(
            "Upload `internet.csv` and `mobile.csv` via sidebar to execute pipeline."
        )
        st.stop()

# Run Data Processing
df_cleaned, logs = process_and_clean_telecom_data(
    df_int, df_mob, missing_strategy=missing_strategy
)
df_growth = compute_country_growth(df_cleaned)
df_stack_anomalies = detect_stack_anomalies(df_cleaned)

latest_year = int(
    df_cleaned.dropna(subset=["internet_users_pct"])["Year"].max()
)

# Header Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Cleaned Records", f"{logs['final_rows']:,}")
kpi2.metric(
    "Outliers Corrected",
    logs["pct_outliers_flagged"] + logs["sub_outliers_corrected"],
)
kpi3.metric(
    "Stack Anomalies (YoY Drops)",
    len(df_stack_anomalies),
    delta="Monotonic Stack Flag",
    delta_color="inverse",
)
kpi4.metric(
    "Top Growth Country",
    f"{df_growth.iloc[0]['Country']}",
    f"+{df_growth.iloc[0]['pct_points_growth']:.1f}% pts",
)

st.markdown("---")

# ==========================================
# 5. INTERACTIVE CHOROPLETH MAP
# ==========================================
st.header(f"🗺️ Global Internet Penetration Map ({latest_year})")
df_recent_map = (
    df_cleaned[df_cleaned["Year"] == latest_year]
    .dropna(subset=["internet_users_pct"])
    .copy()
)

fig_map = px.choropleth(
    df_recent_map,
    locations="Country",
    locationmode="country names",
    color="internet_users_pct",
    hover_name="Country",
    hover_data={
        "internet_users_pct": ":.1f%",
        "mobile_subscriptions": ":.1f",
        "Country": False,
    },
    color_continuous_scale="Viridis",
    labels={"internet_users_pct": "Internet Users (%)"},
)
fig_map.update_layout(
    geo=dict(
        showframe=False, showcoastlines=True, projection_type="natural earth"
    ),
    margin=dict(l=0, r=0, t=10, b=10),
    height=480,
)
st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ==========================================
# 6. COUNTRY INSPECTOR & SINGLE TREND
# ==========================================
st.header("🔍 Country Deep-Dive Inspector")
all_countries = sorted(df_cleaned["Country"].dropna().unique())
selected_country = st.selectbox(
    "Select Country:",
    options=all_countries,
    index=(
        all_countries.index("United States")
        if "United States" in all_countries
        else 0
    ),
)

if selected_country:
    df_country = df_cleaned[df_cleaned["Country"] == selected_country].sort_values(
        "Year"
    )
    c_growth = df_growth[df_growth["Country"] == selected_country]

    if not c_growth.empty:
        cg_row = c_growth.iloc[0]
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric(
            f"Current Penetration ({latest_year})", f"{cg_row['last_pct']:.1f}%"
        )
        stat2.metric(
            f"Start Penetration ({cg_row['first_year']})",
            f"{cg_row['first_pct']:.1f}%",
        )
        stat3.metric(
            "Total Growth (% Pts)",
            f"+{cg_row['pct_points_growth']:.1f}%",
            delta=f"{cg_row['duration_years']} Years Span",
        )
        stat4.metric(
            "3-Yr Queue Avg",
            f"{df_country['internet_pct_3yr_avg'].iloc[-1]:.1f}%",
        )

    fig_single = go.Figure()
    fig_single.add_trace(
        go.Scatter(
            x=df_country["Year"],
            y=df_country["internet_users_pct"],
            mode="lines+markers",
            name="Raw %",
            line=dict(dash="dash", color="#3498db", width=1.5),
        )
    )
    fig_single.add_trace(
        go.Scatter(
            x=df_country["Year"],
            y=df_country["internet_pct_3yr_avg"],
            mode="lines",
            name="3-Yr Queue Rolling Avg",
            line=dict(color="#2ecc71", width=3),
        )
    )
    fig_single.update_layout(
        xaxis_title="Year",
        yaxis_title="Internet Penetration Rate (%)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=30),
    )
    st.plotly_chart(fig_single, use_container_width=True)

st.markdown("---")

# ==========================================
# 7. RANKINGS: LEVEL vs. GROWTH SORTING
# ==========================================
st.header("📊 Comparative Analytics & Sorting Engine")

comp_col1, comp_col2 = st.columns([1, 1])

with comp_col1:
    st.subheader("Multi-Country Trend Comparison")
    multi_selected = st.multiselect(
        "Select Countries for Line Chart:",
        options=all_countries,
        default=[
            c
            for c in ["United Kingdom", "United States", "India", "China"]
            if c in all_countries
        ],
    )
    if multi_selected:
        df_multi = df_cleaned[df_cleaned["Country"].isin(multi_selected)]
        fig_multi = px.line(
            df_multi,
            x="Year",
            y="internet_pct_3yr_avg",
            color="Country",
            title="3-Year Rolling Average Comparison",
            labels={"internet_pct_3yr_avg": "Internet % (3-Yr Avg)"},
        )
        fig_multi.update_layout(height=420)
        st.plotly_chart(fig_multi, use_container_width=True)

with comp_col2:
    st.subheader("Top 5 & Bottom 5 Rankings")

    # Interactive Toggle: Level vs Growth Sorting
    sort_criterion = st.radio(
        "Rank Countries By:",
        options=["Absolute Adoption Level (%)", "Growth (Percentage Points Gained)"],
        horizontal=True,
    )

    if sort_criterion == "Absolute Adoption Level (%)":
        df_rank = (
            df_cleaned[df_cleaned["Year"] == latest_year]
            .dropna(subset=["internet_users_pct"])
            .sort_values("internet_users_pct", ascending=False)
        )

        top_5 = df_rank.head(5)
        bottom_5 = df_rank.tail(5)
        tb_df = pd.concat([top_5, bottom_5]).sort_values(
            "internet_users_pct", ascending=True
        )
        tb_df["Group"] = np.where(
            tb_df["internet_users_pct"] >= 50, "Top 5 Level", "Bottom 5 Level"
        )

        fig_rank = px.bar(
            tb_df,
            x="internet_users_pct",
            y="Country",
            orientation="h",
            color="Group",
            color_discrete_map={
                "Top 5 Level":"#2ecc71",
                "Bottom 5 Level": "#e74c3c",
            },
            text_auto=".1f",
            title=f"Ranked by Level in {latest_year} (%)",
            labels={"internet_users_pct": "Internet Users (%)"},
        )
    else:
        top_5_growth = df_growth.head(5)
        bottom_5_growth = df_growth.tail(5)
        tb_df = pd.concat([top_5_growth, bottom_5_growth]).sort_values(
            "pct_points_growth", ascending=True
        )
        tb_df["Group"] = np.where(
            tb_df["pct_points_growth"] >= 50,
            "Top 5 Growth",
            "Bottom 5 Growth",
        )

        fig_rank = px.bar(
            tb_df,
            x="pct_points_growth",
            y="Country",
            orientation="h",
            color="Group",
            color_discrete_map={
                "Top 5 Growth": "#9b59b6",
                "Bottom 5 Growth": "#e67e22",
            },
            text_auto="+.1f",
            hover_data=["first_year", "last_year", "first_pct", "last_pct"],
            title="Ranked by Total Growth (Last Year % - First Year %)",
            labels={"pct_points_growth": "Percentage Points Gained (% pts)"},
        )

    fig_rank.update_layout(height=420, showlegend=True)
    st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")

# ==========================================
# 8. DATA STRUCTURES IN ACTION: STACK & QUEUE
# ==========================================
st.header("⚡ Data Structures in Action: Stack & Queue Patterns")

ds_tab1, ds_tab2 = st.tabs(
    ["Monotonic Stack Anomaly Detector", "📥 Deque Queue Rolling Window"]
)

with ds_tab1:
    st.subheader("Monotonic Stack YoY Drop Audit")
    st.markdown("""
    **Concept:** Internet adoption should strictly increase or hold steady over time ($\text{Val}_t \ge \text{Val}_{t-1}$). 
    We traverse each country's history in order using a **Monotonic Stack**. 
    When adoption rises or holds steady, the year/value pair is pushed onto the stack. When adoption drops below the stack's top, 
    the top element is popped and flagged as a data drop/outage.
    """)

    if not df_stack_anomalies.empty:
        st.warning(
            f"Detected {len(df_stack_anomalies)} Year-over-Year drops across all country time series:"
        )
        st.dataframe(df_stack_anomalies, use_container_width=True)
    else:
        st.success(
            "Zero monotonic anomalies detected! All country trajectories are strictly non-decreasing."
        )

with ds_tab2:
    st.subheader("Manual Rolling Average with `collections.deque(maxlen=3)`")
    st.markdown("""
    **Concept:** Replaces opaque pandas `.rolling()` operations by explicitly pushing each incoming yearly data point into a 
     fixed-size **FIFO Queue** (`collections.deque(maxlen=3)`).
    
    When a 4th year arrives, `deque` automatically evicts the oldest 1st year (FIFO eviction) and computes `sum(deque) / len(deque)` in $O(1)$ amortized time per element.
    """)

    sample_country = (
        selected_country if selected_country else all_countries[0]
    )
    sample_df = (
        df_cleaned[df_cleaned["Country"] == sample_country]
        .sort_values("Year")
        .head(8)[["Year", "internet_users_pct", "internet_pct_3yr_avg"]]
    )

    st.write(f"**Step-by-Step Deque Queue Walkthrough for `{sample_country}`:**")
    st.dataframe(
        sample_df.rename(
            columns={
                "internet_users_pct": "Raw Input Value (%)",
                "internet_pct_3yr_avg": "Deque Averaged Result (%)",
            }
        ),
        use_container_width=True,
    )

st.markdown("---")

# ==========================================
# 9. EXPORT & AUDIT INSPECTOR
# ==========================================
st.subheader("Dataset & Audit Logs Export")

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    df_cleaned.to_excel(writer, index=False, sheet_name="Cleaned_Data")
    df_growth.to_excel(writer, index=False, sheet_name="Growth_Leaderboard")
    df_stack_anomalies.to_excel(
        writer, index=False, sheet_name="Stack_Anomalies"
    )

st.download_button(
    label="Export Full Audit Workbook (.xlsx)",
    data=buffer.getvalue(),
    file_name="Telecom_Data_Full_Audit.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)