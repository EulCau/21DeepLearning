import random

from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))


def lower(text):
    return text.lower()


def shuffle_sent(text):
    sentences = sent_tokenize(text)
    random.shuffle(sentences)
    return " ".join(sentences)


def shuffle_words_in_sent(text):
    sentences = sent_tokenize(text)
    new_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        random.shuffle(words)
        new_sentences.append(" ".join(words))
    return " ".join(new_sentences)


def drop_stopwords(text):
    words = word_tokenize(text)
    return " ".join(w for w in words if w.lower() not in STOPWORDS)


def drop_random_sentence(text, drop_prob=0.3):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return text
    filtered = [s for s in sentences if random.random() > drop_prob]
    return " ".join(filtered) if filtered else sentences[0]


def get_synonym(word):
    syns = wordnet.synsets(word)
    lemmas = set()
    for s in syns:
        for l in s.lemmas():
            if l.name().lower() != word.lower():
                lemmas.add(l.name().replace('_', ' '))
    return random.choice(list(lemmas)) if lemmas else word


def replace_synonyms(text, ratio=0.2):
    words = word_tokenize(text)
    new_words = []
    for w in words:
        if w.isalpha() and random.random() < ratio:
            new_words.append(get_synonym(w))
        else:
            new_words.append(w)
    return ' '.join(new_words)


def char_noise(text, error_rate=0.05):
    noisy_text = []
    for c in text:
        if random.random() < error_rate:
            op = random.choice(['drop', 'swap', 'repeat'])
            if op == 'drop':
                continue
            elif op == 'swap' and len(noisy_text) > 0:
                noisy_text[-1], c = c, noisy_text[-1]
            elif op == 'repeat':
                noisy_text.append(c)
        noisy_text.append(c)
    return ''.join(noisy_text)


def apply_tta(text, mode="lower"):
    if mode == "lower":
        return lower(text)
    elif mode == "shuffle_sent":
        return shuffle_sent(text)
    elif mode == "shuffle_words":
        return shuffle_words_in_sent(text)
    elif mode == "drop_stopwords":
        return drop_stopwords(text)
    elif mode == "drop_sentence":
        return drop_random_sentence(text)
    elif mode == "replace_synonym":
        return replace_synonyms(text)
    elif mode == "char_noise":
        return char_noise(text)
    else:
        raise ValueError(f"Unknown TTA mode: {mode}")
