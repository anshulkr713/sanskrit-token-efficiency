from pathlib import Path
import os
import pandas as pd
import tiktoken
import sentencepiece as spm
import google.generativeai as genai

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "parallel_gita.csv"
TOKENIZER_DIR = ROOT / "tokenizers"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def load_spm(lang: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    model_path = TOKENIZER_DIR / f"spm_{lang}.model"
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
    sp.load(str(model_path))
    return sp


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run prepare_parallel_gita.py first.")

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    # ---------- GPT tokenizers ----------
    print("Loading GPT tokenizers (cl100k_base and o200k_base) ...")
    gpt_enc = tiktoken.get_encoding("cl100k_base")
    o200k_enc = tiktoken.get_encoding("o200k_base")

    # ---------- SentencePiece tokenizers ----------
    print("Loading SentencePiece tokenizers ...")
    sp_sa = load_spm("sa")
    sp_sa_latn = load_spm("sa_latn")
    sp_hi = load_spm("hi")
    sp_en = load_spm("en")

    def count_spm(text: str, lang: str) -> int:
        if lang == "sa":
            return len(sp_sa.encode(text, out_type=int))
        elif lang == "sa_latn":
            return len(sp_sa_latn.encode(text, out_type=int))
        elif lang == "hi":
            return len(sp_hi.encode(text, out_type=int))
        elif lang == "en":
            return len(sp_en.encode(text, out_type=int))
        else:
            raise ValueError(f"Unknown lang: {lang}")

    # ---------- Gemini tokenizer setup ----------
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    # you can use "gemini-1.5-flash" or "gemini-1.5-pro"
    
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
    def count_gemini_tokens(text: str) -> int:
        """
        Uses Gemini's count_tokens API for the given text.
        """
        # The SDK accepts a plain string as contents
        resp = gemini_model.count_tokens(text)
        # resp.total_tokens is the documented attribute
        return resp.total_tokens

    rows = []
    print("Computing token statistics ...")
    for _, row in df.iterrows():
        text = str(row["text"])
        lang = row["lang"]
        verse_id = row["verse_id"]

        chars = len(text)
        gpt_tokens = len(gpt_enc.encode(text))
        o200k_tokens = len(o200k_enc.encode(text))
        spm_tokens = count_spm(text, lang)

        try:
            gemini_tokens = count_gemini_tokens(text)
        except Exception as e:
            # If something fails (rate limit, network, etc.), you can log -1 or skip
            print(f"Gemini token count error on verse_id={verse_id}, lang={lang}: {e}")
            gemini_tokens = -1

        rows.append({
            "verse_id": verse_id,
            "lang": lang,
            "chars": chars,
            "gpt_tokens": gpt_tokens,
            "o200k_tokens": o200k_tokens,
            "gemini_tokens": gemini_tokens,
            "spm_tokens": spm_tokens,
            "chars_per_gpt_tok": chars / gpt_tokens if gpt_tokens > 0 else None,
            "chars_per_o200k_tok": chars / o200k_tokens if o200k_tokens > 0 else None,
            "chars_per_gemini_tok": chars / gemini_tokens if gemini_tokens > 0 else None,
            "chars_per_spm_tok": chars / spm_tokens if spm_tokens > 0 else None,
        })

    out_df = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / "exp1_results_with_gemini.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    # quick summaries
    print("\n=== GPT (cl100k) tokens per sentence (mean) ===")
    print(out_df.groupby("lang")["gpt_tokens"].mean())

    print("\n=== GPT (o200k) tokens per sentence (mean) ===")
    print(out_df.groupby("lang")["o200k_tokens"].mean())

    print("\n=== Gemini tokens per sentence (mean) ===")
    print(out_df.groupby("lang")["gemini_tokens"].mean())

    print("\n=== SentencePiece tokens per sentence (mean) ===")
    print(out_df.groupby("lang")["spm_tokens"].mean())

    print("\n=== Chars per Gemini token (mean) ===")
    print(out_df.groupby("lang")["chars_per_gemini_tok"].mean())


if __name__ == "__main__":
    main()
