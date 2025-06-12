import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))

def lower(text):
    return text.lower()

def shuffle_sent(text):
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    return " ".join(sentences)

def drop_stopwords(text):
    words = word_tokenize(text)
    return " ".join(w for w in words if w.lower() not in STOPWORDS)

def apply_tta(text, mode='lower'):
    if mode == 'lower':
        return lower(text)
    elif mode == 'shuffle_sent':
        return shuffle_sent(text)
    elif mode == 'drop_stopwords':
        return drop_stopwords(text)
    else:
        raise ValueError(f"Unknown TTA mode: {mode}")
