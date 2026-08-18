import numpy as np
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

# ==========================================
# STEP 1: DATA CLEANING FUNCTION
# ==========================================
COUNTRY_ALIAS_MAP = {
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "GREAT BRITAIN": "United Kingdom",
    "ENGLAND": "United Kingdom",
    "US": "United States",
    "U.S.": "United States",
    "U.S.A.": "United States",
    "USA": "United States",
    "UNITED STATES": "United States",
    "UNITED STATES OF AMERICA": "United States",
    "IND": "India",
    "INDIA": "India",
    "PRC": "China",
    "CHINA": "China",
    "PEOPLES REPUBLIC OF CHINA": "China",
    "GER": "Germany",
    "GERMANY": "Germany",
    "DEUTSCHLAND": "Germany",
    "FRA": "France",
    "FRANCE": "France",
    "BRA": "Brazil",
    "BRAZIL": "Brazil",
    "BRASIL": "Brazil",
    "JPN": "Japan",
    "JAPAN": "Japan",
    "CAN": "Canada",
    "CANADA": "Canada",
    "AUS": "Australia",
    "AUSTRALIA": "Australia",
    "RSA": "South Africa",
    "SOUTH AFRICA": "South Africa",
    "MEX": "Mexico",
    "MEXICO": "Mexico",
    "ITA": "Italy",
    "ITALY": "Italy",
    "ESP": "Spain",
    "SPAIN": "Spain",
    "ROK": "South Korea",
    "KOREA": "South Korea",
    "SOUTH KOREA": "South Korea",
    "RUS": "Russia",
    "RUSSIA": "Russia",
    "RUSSIAN FEDERATION": "Russia",
}


def clean_dataframe(df: pd.DataFrame, metric_col_name: str) -> pd.DataFrame:
    """Cleans duplicate rows, normalizes country names, parses numbers/years,

    and fixes impossible values.
    """
    df_clean = df.copy()

    # 1. Drop exact duplicates
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2. Normalize country names
    def fix_country(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s in COUNTRY_ALIAS_MAP:
            return COUNTRY_ALIAS_MAP[s]
        s_nodots = s.replace(".", "")
        if s_nodots in COUNTRY_ALIAS_MAP:
            return COUNTRY_ALIAS_MAP[s_nodots]
        return str(val).strip().title()

    if "country" in df_clean.columns:
        df_clean["country"] = df_clean["country"].apply(fix_country)
    elif "Country" in df_clean.columns:
        df_clean["Country"] = df_clean["Country"].apply(fix_country)

    # 3. Clean year column
    year_col = "year" if "year" in df_clean.columns else "Year"
    if year_col in df_clean.columns:
        df_clean[year_col] = pd.to_numeric(
            df_clean[year_col], errors="coerce"
        ).astype("Int64")

    # 4. Clean metric column (strip '%', convert strings to numeric)
    if metric_col_name in df_clean.columns:
        df_clean[metric_col_name] = (
            df_clean[metric_col_name]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df_clean[metric_col_name] = pd.to_numeric(
            df_clean[metric_col_name], errors="coerce"
        )

        # Catch & fix impossible values
        if "percent" in metric_col_name.lower() or "%" in metric_col_name:
            df_clean.loc[
                (df_clean[metric_col_name] < 0)
                | (df_clean[metric_col_name] > 100),
                metric_col_name,
            ] = np.nan
        else:
            df_clean[metric_col_name] = df_clean[metric_col_name].apply(
                lambda x: abs(x) if pd.notna(x) and x < 0 else x
            )

    # 5. Remove duplicates created after country name standardization
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 6. Impute missing values per country group
    country_col = "country" if "country" in df_clean.columns else "Country"
    if country_col in df_clean.columns and year_col in df_clean.columns:
        df_clean = df_clean.sort_values(
            by=[country_col, year_col]
        ).reset_index(drop=True)
        df_clean[metric_col_name] = (
            df_clean.groupby(country_col)[metric_col_name]
            .transform(lambda grp: grp.ffill().bfill())
        )

    return df_clean


# ==========================================
# STEP 2: EXECUTE CLEANING & EXPORT TO EXCEL
# ==========================================
# Read raw CSVs
df_internet = pd.read_csv("internet.csv")
df_mobile = pd.read_csv("mobile.csv")

# Clean both datasets
df_internet_clean = clean_dataframe(
    df_internet, metric_col_name="internet_users_pct"
)
df_mobile_clean = clean_dataframe(
    df_mobile, metric_col_name="mobile_subscriptions"
)

# Merge cleaned datasets side-by-side
df_merged = pd.merge(
    df_internet_clean,
    df_mobile_clean,
    on=["country", "year"],
    how="outer",
    suffixes=("_internet", "_mobile"),
)

# Export to Excel
with pd.ExcelWriter("Cleaned_Telecom_Data.xlsx", engine="openpyxl") as writer:
    df_merged.to_excel(
        writer, sheet_name="Cleaned & Merged Data", index=False
    )
    df_internet_clean.to_excel(writer, sheet_name="Cleaned Internet", index=False)
    df_mobile_clean.to_excel(writer, sheet_name="Cleaned Mobile", index=False)

print("Data successfully cleaned and saved to Cleaned_Telecom_Data.xlsx")