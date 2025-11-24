from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Change this if you used a different filename
RES_PATH = ROOT / "analysis" / "exp1_results_with_translit.csv"

def main():
    if not RES_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RES_PATH}")

    df = pd.read_csv(RES_PATH)

    print("Languages present and counts:")
    print(df["lang"].value_counts(), "\n")

    # --- GPT token stats ---
    print("=== GPT tokens per sentence ===")
    print(df.groupby("lang")["gpt_tokens"].agg(["mean", "std", "min", "max"]))
    print()

    # --- SPM token stats ---
    print("=== SentencePiece tokens per sentence ===")
    print(df.groupby("lang")["spm_tokens"].agg(["mean", "std", "min", "max"]))
    print()

    # --- Chars per token (GPT) ---
    print("=== Characters per GPT token (mean) ===")
    print(df.groupby("lang")["chars_per_gpt_tok"].mean())
    print()

    # --- Chars per token (SPM) ---
    print("=== Characters per SentencePiece token (mean) ===")
    print(df.groupby("lang")["chars_per_spm_tok"].mean())
    print()

    # --- Direct comparison: Sanskrit Devanagari vs Latin ---
    subset = df[df["lang"].isin(["sa", "sa_latn"])]

    print("=== DIRECT COMPARISON: sa vs sa_latn (GPT tokens) ===")
    print(subset.groupby("lang")["gpt_tokens"].agg(["mean", "std", "min", "max"]))
    print()

    print("=== DIRECT COMPARISON: sa vs sa_latn (SPM tokens) ===")
    print(subset.groupby("lang")["spm_tokens"].agg(["mean", "std", "min", "max"]))
    print()

    print("=== DIRECT COMPARISON: sa vs sa_latn (chars per GPT token) ===")
    print(subset.groupby("lang")["chars_per_gpt_tok"].mean())
    print()

    print("=== DIRECT COMPARISON: sa vs sa_latn (chars per SPM token) ===")
    print(subset.groupby("lang")["chars_per_spm_tok"].mean())
    print()

if __name__ == "__main__":
    main()
