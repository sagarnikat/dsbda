# Step 1: Import Libraries
import re
import math
import pandas as pd

from collections import Counter
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Create Sample Document
text = """
Data Analytics is one of the most important technologies in modern computer science.
It is used in machine learning, artificial intelligence, business analysis,
data visualization and prediction systems.
"""

# Step 3: Display Original Text
print("Original Text:\n")
print(text)

# --------------------------------------------------
# Step 4: Tokenization
# --------------------------------------------------

tokens = re.findall(r'\b\w+\b', text.lower())

print("\nTokens:\n")
print(tokens)

# --------------------------------------------------
# Step 5: Manual POS Tagging (Offline Simple Version)
# --------------------------------------------------

# Simple rule-based POS tagging
# (Not accurate like NLTK model but works offline)

pos_tags = []

for word in tokens:

    if word.endswith("ing"):
        tag = "VBG"      # Verb Gerund

    elif word.endswith("ion"):
        tag = "NN"       # Noun

    elif word.endswith("ed"):
        tag = "VBD"      # Verb Past

    else:
        tag = "NN"       # Default Noun

    pos_tags.append((word, tag))

print("\nPOS Tags:\n")
print(pos_tags)

# --------------------------------------------------
# Step 6: Stopword Removal
# --------------------------------------------------

stop_words = {
    'is', 'one', 'of', 'the', 'most', 'in',
    'it', 'and', 'on', 'a', 'an', 'to'
}

filtered_words = []

for word in tokens:

    if word not in stop_words:

        filtered_words.append(word)

print("\nAfter Stopword Removal:\n")
print(filtered_words)

# --------------------------------------------------
# Step 7: Stemming
# --------------------------------------------------

stemmer = PorterStemmer()

stemmed_words = []

for word in filtered_words:

    stemmed_words.append(stemmer.stem(word))

print("\nStemmed Words:\n")
print(stemmed_words)

# --------------------------------------------------
# Step 8: Lemmatization (Manual Offline Version)
# --------------------------------------------------

# Simple manual lemmatization dictionary

lemma_dict = {
    "technologies": "technology",
    "systems": "system",
    "learning": "learn",
    "analytics": "analytic"
}

lemmatized_words = []

for word in filtered_words:

    if word in lemma_dict:
        lemmatized_words.append(lemma_dict[word])

    else:
        lemmatized_words.append(word)

print("\nLemmatized Words:\n")
print(lemmatized_words)

# --------------------------------------------------
# Step 9: Join Processed Words
# --------------------------------------------------

processed_text = " ".join(stemmed_words)

print("\nProcessed Text:\n")
print(processed_text)

# --------------------------------------------------
# Step 10: Calculate Term Frequency (TF)
# --------------------------------------------------

word_count = Counter(stemmed_words)

total_words = len(stemmed_words)

tf_dict = {}

for word, count in word_count.items():

    tf_dict[word] = count / total_words

print("\nTerm Frequency (TF):\n")

for word, tf in tf_dict.items():

    print(word, ":", round(tf, 3))

# --------------------------------------------------
# Step 11: Calculate Inverse Document Frequency (IDF)
# --------------------------------------------------

# Using single document example

idf_dict = {}

for word in word_count.keys():

    idf = math.log((1 + 1) / (1 + 1)) + 1

    idf_dict[word] = idf

print("\nInverse Document Frequency (IDF):\n")

for word, idf in idf_dict.items():

    print(word, ":", round(idf, 3))

# --------------------------------------------------
# Step 12: TF-IDF Vectorization
# --------------------------------------------------

vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform([processed_text])

# --------------------------------------------------
# Step 13: Feature Names
# --------------------------------------------------

print("\nFeature Names:\n")
print(vectorizer.get_feature_names_out())

# --------------------------------------------------
# Step 14: TF-IDF Matrix
# --------------------------------------------------

print("\nTF-IDF Matrix:\n")
print(tfidf.toarray())

# --------------------------------------------------
# Step 15: TF-IDF DataFrame
# --------------------------------------------------

tfidf_df = pd.DataFrame(
    tfidf.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nTF-IDF DataFrame:\n")
print(tfidf_df)

# --------------------------------------------------
# Step 16: Final Observation
# --------------------------------------------------

print("\nText preprocessing completed successfully.")