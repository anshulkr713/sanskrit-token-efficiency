from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "analysis" / "exp1_results_with_gemini.csv"

def main():
    df = pd.read_csv(RES_PATH)

    print("Languages present:")
    print(df["lang"].value_counts(), "\n")

    # 1) Mean chars per token (this is your token efficiency) for each tokenizer
    print("=== Mean characters per token (GPT cl100k) ===")
    print(df.groupby("lang")["chars_per_gpt_tok"].mean(), "\n")

    print("=== Mean characters per token (GPT o200k) ===")
    print(df.groupby("lang")["chars_per_o200k_tok"].mean(), "\n")

    print("=== Mean characters per token (Gemini) ===")
    print(df.groupby("lang")["chars_per_gemini_tok"].mean(), "\n")

    print("=== Mean characters per token (SPM) ===")
    print(df.groupby("lang")["chars_per_spm_tok"].mean(), "\n")

    # 2) Optional: tokens-per-char (cost per information unit)
    df["gpt_tokens_per_char"] = df["gpt_tokens"] / df["chars"]
    df["o200k_tokens_per_char"] = df["o200k_tokens"] / df["chars"]
    df["gemini_tokens_per_char"] = df["gemini_tokens"] / df["chars"]
    df["spm_tokens_per_char"] = df["spm_tokens"] / df["chars"]

    print("=== GPT tokens per character (mean) ===")
    print(df.groupby("lang")["gpt_tokens_per_char"].mean(), "\n")

    print("=== SPM tokens per character (mean) ===")
    print(df.groupby("lang")["spm_tokens_per_char"].mean(), "\n")

if __name__ == "__main__":
    main()
