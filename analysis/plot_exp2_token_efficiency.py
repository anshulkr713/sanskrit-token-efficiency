from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES_PATH = ROOT / "analysis" / "exp2_commentary_results.csv"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RES_PATH)

    # For clarity, create a human-readable label for each (lang, segment_type)
    def label_row(row):
        if row["segment_type"] == "sa_verse":
            return "SA verse"
        if row["segment_type"] == "en_trans":
            return "EN translation"
        if row["segment_type"] == "en_mean":
            return "EN commentary"
        if row["segment_type"] == "hi_trans":
            return "HI translation"
        if row["segment_type"] == "hi_mean":
            return "HI commentary"
        return f"{row['lang']}_{row['segment_type']}"

    df["segment_label"] = df.apply(label_row, axis=1)

    # We'll preserve this order on the x-axis
    seg_order = [
        "SA verse",
        "EN translation",
        "HI translation",
        "EN commentary",
        "HI commentary",
    ]
    df = df[df["segment_label"].isin(seg_order)]

    # 1) Mean tokens per segment label per tokenizer
    token_cols = ["gpt_tokens", "o200k_tokens", "gemini_tokens", "spm_tokens"]
    mean_tokens = (
        df.groupby("segment_label")[token_cols]
        .mean()
        .loc[seg_order]  # enforce order
        .rename(columns={
            "gpt_tokens": "GPT-cl100k",
            "o200k_tokens": "GPT-o200k",
            "gemini_tokens": "Gemini",
            "spm_tokens": "SPM",
        })
    )

    plt.figure(figsize=(10, 5))
    x = range(len(seg_order))
    width = 0.2

    for i, col in enumerate(mean_tokens.columns):
        plt.bar(
            [p + i * width for p in x],
            mean_tokens[col].values,
            width=width,
            label=col,
        )

    plt.xticks([p + 1.5 * width for p in x], seg_order, rotation=20, ha="right")
    plt.ylabel("Mean tokens per segment")
    plt.title("Tokens per Segment Type by Tokenizer (Exp2)")
    plt.legend()
    plt.tight_layout()
    out_path1 = FIG_DIR / "exp2_mean_tokens_per_segment.png"
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved {out_path1}")

    # 2) Mean characters per token per segment label per tokenizer
    chars_per_cols = [
        "chars_per_gpt_tok",
        "chars_per_o200k_tok",
        "chars_per_gemini_tok",
        "chars_per_spm_tok",
    ]
    mean_chars_per = (
        df.groupby("segment_label")[chars_per_cols]
        .mean()
        .loc[seg_order]
        .rename(columns={
            "chars_per_gpt_tok": "GPT-cl100k",
            "chars_per_o200k_tok": "GPT-o200k",
            "chars_per_gemini_tok": "Gemini",
            "chars_per_spm_tok": "SPM",
        })
    )

    plt.figure(figsize=(10, 5))
    for i, col in enumerate(mean_chars_per.columns):
        plt.bar(
            [p + i * width for p in x],
            mean_chars_per[col].values,
            width=width,
            label=col,
        )

    plt.xticks([p + 1.5 * width for p in x], seg_order, rotation=20, ha="right")
    plt.ylabel("Mean characters per token")
    plt.title("Token Efficiency (Chars per Token) by Segment Type and Tokenizer (Exp2)")
    plt.legend()
    plt.tight_layout()
    out_path2 = FIG_DIR / "exp2_chars_per_token_per_segment.png"
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
