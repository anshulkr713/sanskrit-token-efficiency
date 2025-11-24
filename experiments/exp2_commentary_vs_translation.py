from pathlib import Path
import os
import pandas as pd
import tiktoken
import sentencepiece as spm
import google.generativeai as genai
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "bhagavad_gita.csv"
TOKENIZER_DIR = ROOT / "tokenizers"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = ANALYSIS_DIR / "exp2_commentary_results.csv"


def load_spm(lang: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    model_path = TOKENIZER_DIR / f"spm_{lang}.model"
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
    sp.load(str(model_path))
    return sp


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"{RAW_PATH} not found. Put bhagavad_gita.csv in data/raw/")

    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)

    required_cols = [
        "verse_number",
        "verse_in_sanskrit",
        "translation_in_english",
        "meaning_in_english",
        "translation_in_hindi",
        "meaning_in_hindi",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    print("Loading GPT tokenizers (cl100k_base and o200k_base) ...")
    gpt_enc = tiktoken.get_encoding("cl100k_base")
    o200k_enc = tiktoken.get_encoding("o200k_base")

    print("Loading SentencePiece tokenizers (sa, hi, en) ...")
    sp_sa = load_spm("sa")
    sp_hi = load_spm("hi")
    sp_en = load_spm("en")

    def count_spm(text: str, lang: str) -> int:
        if lang == "sa":
            return len(sp_sa.encode(text, out_type=int))
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
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

    def count_gemini_tokens(text: str) -> int:
        try:
            resp = gemini_model.count_tokens(text)
            
            # SAFETY SLEEP: 0.04s delay caps us at ~1500 RPM (well below the 3000 limit)
            time.sleep(0.04) 
            
            return resp.total_tokens
        except exceptions.ResourceExhausted:
            print(f"\nRate limit hit! Cooling down for 10 seconds...")
            time.sleep(10)
            # Retry once recursively
            try:
                resp = gemini_model.count_tokens(text)
                return resp.total_tokens
            except:
                return -1
        except Exception as e:
            # Catch 404s or other random network blips
            print(f"Error: {e}")
            return -1

    rows = []
    print("Computing token statistics for verse / translation / meaning ...")

    for _, r in tqdm(df.iterrows(), total=len(df)):
        verse_id = str(r["verse_number"]).strip()

        sa_verse = r.get("verse_in_sanskrit", "")
        en_trans = r.get("translation_in_english", "")
        en_mean = r.get("meaning_in_english", "")
        hi_trans = r.get("translation_in_hindi", "")
        hi_mean = r.get("meaning_in_hindi", "")

        # helper to add one segment
        def add_segment(text, lang, segment_type):
            if pd.isna(text):
                return
            text = str(text).strip()
            if not text:
                return

            chars = len(text)
            gpt_tokens = len(gpt_enc.encode(text))
            o200k_tokens = len(o200k_enc.encode(text))
            spm_tokens = count_spm(text, lang)

            try:
                gemini_tokens = count_gemini_tokens(text)
            except Exception as e:
                print(f"Gemini token count error on verse_id={verse_id}, lang={lang}, segment={segment_type}: {e}")
                gemini_tokens = -1

            rows.append({
                "verse_id": verse_id,
                "lang": lang,
                "segment_type": segment_type,
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

        # Sanskrit root verse (compact)
        add_segment(sa_verse, "sa", "sa_verse")

        # English translation + meaning
        add_segment(en_trans, "en", "en_trans")
        add_segment(en_mean, "en", "en_mean")

        # Hindi translation + meaning
        add_segment(hi_trans, "hi", "hi_trans")
        add_segment(hi_mean, "hi", "hi_mean")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved results to {OUT_PATH}")

    # quick summaries
    print("\n=== GPT (cl100k) tokens per segment type (mean) ===")
    print(out_df.groupby(["lang", "segment_type"])["gpt_tokens"].mean())

    print("\n=== GPT (o200k) tokens per segment type (mean) ===")
    print(out_df.groupby(["lang", "segment_type"])["o200k_tokens"].mean())

    print("\n=== Gemini tokens per segment type (mean) ===")
    print(out_df.groupby(["lang", "segment_type"])["gemini_tokens"].mean())

    print("\n=== SPM tokens per segment type (mean) ===")
    print(out_df.groupby(["lang", "segment_type"])["spm_tokens"].mean())

    print("\n=== Chars per Gemini token (mean) ===")
    print(out_df.groupby(["lang", "segment_type"])["chars_per_gemini_tok"].mean())


if __name__ == "__main__":
    main()
