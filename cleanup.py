import pandas as pd
import re
from geotext import GeoText

# Input and output file names
INPUT_CSV = "clean_autocomplete_technologies.csv"
OUTPUT_CSV = "output.csv"

# Load CSV
df = pd.read_csv(INPUT_CSV)

# Ensure required columns exist
required_columns = {"technology_name", "category", "description"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")


def remove_parentheses_text(text):
    """
    Remove anything inside parentheses including the parentheses.
    Example:
    'React (JavaScript library)' -> 'React'
    """
    if pd.isna(text):
        return text

    # Remove text inside ()
    text = re.sub(r"\([^)]*\)", "", str(text))

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def contains_city_or_country(text):
    """
    Detect city or country names using GeoText.
    """
    if pd.isna(text):
        return False

    places = GeoText(str(text))

    return bool(places.cities or places.countries)


# Work only on rows where category == 'Framework'
framework_mask = df["category"].astype(str).str.lower() == "framework"

# Step 1: Remove text inside parentheses for framework rows
df.loc[framework_mask, "description"] = (
    df.loc[framework_mask, "description"]
    .apply(remove_parentheses_text)
)

# Step 2: Remove framework rows containing city/country names
remove_mask = (
    framework_mask
    & df["description"].apply(contains_city_or_country)
)

filtered_df = df[~remove_mask].copy()

# Save result
filtered_df.to_csv(OUTPUT_CSV, index=False)

print(f"Filtered CSV saved to: {OUTPUT_CSV}")
