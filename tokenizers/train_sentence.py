from pathlib import Path
import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]  # from tokenizers/ up to project root
PROC_PATH = ROOT / "data" / "processed" / "parallel_gita.csv"
OUT_DIR = ROOT / "tokenizers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_lang_corpus(lang: str, out_path: Path):
    df = pd.read_csv(PROC_PATH)
    texts = df[df["lang"] == lang]["text"].astype(str).tolist()
    print(f"{lang}: {len(texts)} lines for corpus")
    with out_path.open("w", encoding="utf-8") as f:
        for t in texts:
            t = t.replace("\n", " ")
            f.write(t + "\n")

def train_spm(lang: str, vocab_size: int = 8000):
    corpus_path = OUT_DIR / f"corpus_{lang}.txt"
    write_lang_corpus(lang, corpus_path)

    model_prefix = OUT_DIR / f"spm_{lang}"
    print(f"Training SentencePiece for {lang} ...")
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe"
    )
    print(f"Saved {model_prefix}.model and .vocab")

def main():
    if not PROC_PATH.exists():
        raise FileNotFoundError(f"{PROC_PATH} not found. Run prepare_parallel_gita.py first.")

    # now include sa_latn as a separate 'language'
    for lang in ["sa", "sa_latn", "hi", "en"]:
        train_spm(lang, vocab_size=8000)

if __name__ == "__main__":
    main()
