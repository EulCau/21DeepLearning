import nltk
import spacy
import textstat
import string
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))


def stylometry_features(text):
	# Token level features
	words = nltk.word_tokenize(text)
	num_words = len(words)
	num_unique_words = len(set(words))
	lexical_diversity = num_unique_words / (num_words + 1e-5)

	avg_word_length = sum(len(w) for w in words) / (num_words + 1e-5)
	num_stopwords = sum(1 for w in words if w.lower() in stop_words)
	stopword_ratio = num_stopwords / (num_words + 1e-5)

	# Sentence level features
	sentences = nltk.sent_tokenize(text)
	num_sentences = len(sentences)
	avg_sentence_length = num_words / (num_sentences + 1e-5)

	# Punctuation
	punctuation_count = sum(1 for c in text if c in string.punctuation)
	punctuation_ratio = punctuation_count / (len(text) + 1e-5)

	# Readability
	flesch_reading_ease = textstat.flesch_reading_ease(text)

	# POS tag ratio
	pos_tags = nltk.pos_tag(words)
	num_nouns = sum(1 for word, pos in pos_tags if pos.startswith('NN'))
	num_verbs = sum(1 for word, pos in pos_tags if pos.startswith('VB'))
	num_adjs = sum(1 for word, pos in pos_tags if pos.startswith('JJ'))
	num_advs = sum(1 for word, pos in pos_tags if pos.startswith('RB'))

	noun_ratio = num_nouns / (num_words + 1e-5)
	verb_ratio = num_verbs / (num_words + 1e-5)
	adj_ratio = num_adjs / (num_words + 1e-5)
	adv_ratio = num_advs / (num_words + 1e-5)

	# Return feature vector
	return [
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
