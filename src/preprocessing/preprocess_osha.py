import pandas as pd
from pathlib import Path


def clean_value(value):
    """Convert a dataframe value into a clean string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def convert_binary(value, positive_text, negative_text):
    """Convert OSHA binary values into readable text."""
    value = str(value).strip().lower()

    if value in {"1", "1.0", "yes", "y", "true"}:
        return positive_text

    return negative_text


def build_model_text(row):
    """
    Combine the OSHA narrative and selected structured
    fields into a single text representation for NLP.
    """

    narrative = clean_value(row["Final Narrative"])

    event = clean_value(row["EventTitle"])

    nature = clean_value(row["NatureTitle"])

    body = clean_value(row["Part of Body Title"])

    hospitalized = convert_binary(
        row["Hospitalized"],
        "hospitalized",
        "not hospitalized"
    )

    amputation = convert_binary(
        row["Amputation"],
        "amputation occurred",
        "no amputation"
    )

    structured_text = (
        f"Event type: {event}. "
        f"Nature of injury: {nature}. "
        f"Part of body: {body}. "
        f"Hospitalization status: {hospitalized}. "
        f"Amputation status: {amputation}."
    )

    return f"{narrative} {structured_text}"


def preprocess_osha(input_path, output_path):
    """Load OSHA data, create model_text, and save processed data."""

    print("Loading OSHA dataset...")

    df = pd.read_csv(input_path, low_memory=False)

    print(f"Loaded {len(df):,} records.")

    # Create the NLP input text
    df["model_text"] = df.apply(build_model_text, axis=1)

    # Save processed dataset
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")


if __name__ == "__main__":

    input_file = Path(
        "data/raw/January2015toNovember2025.csv"
    )

    output_file = Path(
        "data/processed/osha_processed.csv"
    )

    preprocess_osha(
        input_file,
        output_file
    )