from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "analysis" / "exp2_commentary_results.csv"

def main():
    df = pd.read_csv(RES_PATH)

    print("Segment types present:")
    print(df["segment_type"].value_counts(), "\n")

    # Efficiency per (lang, segment_type)
    print("=== Mean chars per token (GPT cl100k) ===")
    print(df.groupby(["lang", "segment_type"])["chars_per_gpt_tok"].mean(), "\n")

    print("=== Mean chars per token (GPT o200k) ===")
    print(df.groupby(["lang", "segment_type"])["chars_per_o200k_tok"].mean(), "\n")

    print("=== Mean chars per token (Gemini) ===")
    print(df.groupby(["lang", "segment_type"])["chars_per_gemini_tok"].mean(), "\n")

    print("=== Mean chars per token (SPM) ===")
    print(df.groupby(["lang", "segment_type"])["chars_per_spm_tok"].mean(), "\n")

    # Optional: add tokens-per-char metrics
    for col in ["gpt_tokens", "o200k_tokens", "gemini_tokens", "spm_tokens"]:
        df[col + "_per_char"] = df[col] / df["chars"]

    print("=== SPM tokens per character (mean) by segment ===")
    print(df.groupby(["lang", "segment_type"])["spm_tokens_per_char"].mean(), "\n")

if __name__ == "__main__":
    main()
