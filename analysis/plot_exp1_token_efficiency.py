from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "analysis" / "exp1_results_with_gemini.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RES_PATH)

    # We’ll focus on these tokenizers
    token_cols = ["gpt_tokens", "o200k_tokens", "gemini_tokens", "spm_tokens"]
    chars_per_cols = [
        "chars_per_gpt_tok",
        "chars_per_o200k_tok",
        "chars_per_gemini_tok",
        "chars_per_spm_tok",
    ]
    tokenizer_labels = ["GPT-cl100k", "GPT-o200k", "Gemini", "SPM"]

    # 1) Mean tokens per sentence per language
    mean_tokens = (
        df.groupby("lang")[token_cols]
        .mean()
        .rename(columns={
            "gpt_tokens": "GPT-cl100k",
            "o200k_tokens": "GPT-o200k",
            "gemini_tokens": "Gemini",
            "spm_tokens": "SPM",
        })
    )

    # Sort languages in a sensible order
    lang_order = ["en", "hi", "sa", "sa_latn"]
    mean_tokens = mean_tokens.reindex(lang_order)

    # Plot: Mean tokens per sentence
    plt.figure(figsize=(8, 5))
    x = range(len(lang_order))
    width = 0.2

    for i, col in enumerate(mean_tokens.columns):
        plt.bar(
            [p + i * width for p in x],
            mean_tokens[col].values,
            width=width,
            label=col,
        )

    plt.xticks([p + 1.5 * width for p in x], ["EN", "HI", "SA", "SA_latn"])
    plt.ylabel("Mean tokens per verse")
    plt.title("Mean Tokens per Verse by Language and Tokenizer (Exp1)")
    plt.legend()
    plt.tight_layout()
    out_path1 = FIG_DIR / "exp1_mean_tokens_per_language.png"
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved {out_path1}")

    # 2) Mean chars per token (token efficiency)
    mean_chars_per = (
        df.groupby("lang")[chars_per_cols]
        .mean()
        .rename(columns={
            "chars_per_gpt_tok": "GPT-cl100k",
            "chars_per_o200k_tok": "GPT-o200k",
            "chars_per_gemini_tok": "Gemini",
            "chars_per_spm_tok": "SPM",
        })
    )
    mean_chars_per = mean_chars_per.reindex(lang_order)

    plt.figure(figsize=(8, 5))
    for i, col in enumerate(mean_chars_per.columns):
        plt.bar(
            [p + i * width for p in x],
            mean_chars_per[col].values,
            width=width,
            label=col,
        )

    plt.xticks([p + 1.5 * width for p in x], ["EN", "HI", "SA", "SA_latn"])
    plt.ylabel("Mean characters per token")
    plt.title("Token Efficiency (Chars per Token) by Language and Tokenizer (Exp1)")
    plt.legend()
    plt.tight_layout()
    out_path2 = FIG_DIR / "exp1_chars_per_token_per_language.png"
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
