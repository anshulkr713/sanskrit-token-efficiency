from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "analysis" / "exp2_commentary_results.csv"

def main():
    if not RES_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RES_PATH}")

    df = pd.read_csv(RES_PATH)

    print("Segment types present:")
    print(df["segment_type"].value_counts(), "\n")

    print("=== GPT tokens per segment (mean, std) ===")
    print(df.groupby(["lang", "segment_type"])["gpt_tokens"].agg(["mean", "std"]))
    print()

    print("=== SPM tokens per segment (mean, std) ===")
    print(df.groupby(["lang", "segment_type"])["spm_tokens"].agg(["mean", "std"]))
    print()

    # Ratios: how much larger is commentary vs Sanskrit verse (per verse)
    # For this, we pivot by verse_id
    pivot = df.pivot_table(
        index="verse_id",
        columns="segment_type",
        values="spm_tokens",  # you can change to gpt_tokens if you like
        aggfunc="first"
    )

    # You might not have all columns for all verses; that's okay.
    # Compute some interesting ratios where possible:
    for col in ["en_trans", "en_mean", "hi_trans", "hi_mean"]:
        if col in pivot.columns and "sa_verse" in pivot.columns:
            ratio_col = f"{col}_vs_sa_ratio"
            pivot[ratio_col] = pivot[col] / pivot["sa_verse"]
            print(f"\nAverage SPM token ratio {col} / sa_verse:")
            print(pivot[ratio_col].mean())

if __name__ == "__main__":
    main()
