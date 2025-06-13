import json
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path

import nltk
import spacy
import textstat
import string
from nltk.corpus import stopwords

# 初始化全局资源（每个子进程都调用一次）
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def extract_features(text):
    # ========== Stylometric Features ==========
    doc = nlp(text)

    words = [token.text for token in doc if token.is_alpha]
    num_words = len(words)
    num_unique_words = len(set(words))
    lexical_diversity = num_unique_words / (num_words + 1e-5)

    avg_word_length = sum(len(w) for w in words) / (num_words + 1e-5)
    num_stopwords = sum(1 for w in words if w.lower() in stop_words)
    stopword_ratio = num_stopwords / (num_words + 1e-5)

    num_sentences = len(list(doc.sents))
    avg_sentence_length = num_words / (num_sentences + 1e-5)

    punctuation_count = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = punctuation_count / (len(text) + 1e-5)

    try:
        # noinspection PyUnresolvedReferences
        flesch_reading_ease = textstat.flesch_reading_ease(text)
    except:
        flesch_reading_ease = 50.0  # fallback

    num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
    num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
    num_adjs = sum(1 for token in doc if token.pos_ == "ADJ")
    num_advs = sum(1 for token in doc if token.pos_ == "ADV")

    noun_ratio = num_nouns / (num_words + 1e-5)
    verb_ratio = num_verbs / (num_words + 1e-5)
    adj_ratio = num_adjs / (num_words + 1e-5)
    adv_ratio = num_advs / (num_words + 1e-5)

    styl_feats = [
        lexical_diversity,
        avg_word_length,
        stopword_ratio,
        avg_sentence_length,
        punctuation_ratio,
        flesch_reading_ease,
        noun_ratio,
        verb_ratio,
        adj_ratio,
        adv_ratio
    ]

    # ========== Simple Features (补充部分) ==========
    length = len(text)
    num_hashes = text.count('#')
    num_urls = text.count("http")
    num_lists = text.count("\n- ") + text.count("\n1.") + text.count("\n2.")

    simple_feats = [length, num_hashes, num_urls, num_lists]

    # ========== 合并并返回 ==========
    return simple_feats + styl_feats  # 共 14 个特征


def process_line(line):
    try:
        item = json.loads(line)
        item["features"] = extract_features(item["text"])
        return json.dumps(item, ensure_ascii=False)
    except Exception:
        return None  # 可加入日志记录

def main(input_path, output_path, num_workers=4):
    with open(input_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        with open(output_path, 'w', encoding="utf-8") as out_f:
            for result in tqdm(pool.imap(process_line, lines, chunksize=32), total=len(lines)):
                if result:
                    out_f.write(result + '\n')

def init_worker():
    # 每个进程加载 spaCy 和 NLTK 数据
    global nlp, stop_words
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="train.jsonl")
    parser.add_argument("--output", type=Path, default="cached_train.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger_eng")

    print(f"Processing {args.input} -> {args.output} with {args.workers} workers")
    main(args.input, args.output, num_workers=args.workers)
