from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw" / "bhagavad_gita.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "parallel_gita.csv"

def main():
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)

    required_cols = [
        "verse_number",
        "verse_in_sanskrit",
        "sanskrit_verse_transliteration",
        "translation_in_hindi",
        "translation_in_english",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    rows = []
    for _, r in df.iterrows():
        vid = str(r["verse_number"]).strip()

        sa = str(r["verse_in_sanskrit"]) if not pd.isna(r["verse_in_sanskrit"]) else ""
        sa_latn = (
            str(r["sanskrit_verse_transliteration"])
            if not pd.isna(r["sanskrit_verse_transliteration"])
            else ""
        )
        hi = str(r["translation_in_hindi"]) if not pd.isna(r["translation_in_hindi"]) else ""
        en = str(r["translation_in_english"]) if not pd.isna(r["translation_in_english"]) else ""

        # skip if any critical piece missing
        if not sa or not hi or not en:
            continue

        # Sanskrit (Devanagari)
        rows.append({"verse_id": vid, "lang": "sa", "text": sa})

        # Sanskrit transliteration (Latin) – only if present
        if sa_latn:
            rows.append({"verse_id": vid, "lang": "sa_latn", "text": sa_latn})

        # Hindi & English
        rows.append({"verse_id": vid, "lang": "hi", "text": hi})
        rows.append({"verse_id": vid, "lang": "en", "text": en})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out_df)} rows to {OUT_PATH}")
    print("Languages present:", out_df['lang'].value_counts())

if __name__ == "__main__":
    main()
