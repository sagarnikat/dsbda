# Step 1: Import Libraries
import re
import pandas as pd

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

# Step 4: Tokenization (Without NLTK punkt)
tokens = re.findall(r'\b\w+\b', text.lower())

print("\nTokens:\n")
print(tokens)

# Step 5: Manual Stopwords List
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

# Step 6: Stemming
stemmer = PorterStemmer()

stemmed_words = []

for word in filtered_words:

    stemmed_words.append(stemmer.stem(word))

print("\nStemmed Words:\n")
print(stemmed_words)

# Step 7: Join Words Again
processed_text = " ".join(stemmed_words)

print("\nProcessed Text:\n")
print(processed_text)

# Step 8: TF-IDF Vectorization
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform([processed_text])

# Step 9: Display Feature Names
print("\nFeature Names:\n")
print(vectorizer.get_feature_names_out())

# Step 10: Display TF-IDF Matrix
print("\nTF-IDF Matrix:\n")
print(tfidf.toarray())

# Step 11: Create TF-IDF DataFrame
tfidf_df = pd.DataFrame(
    tfidf.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nTF-IDF DataFrame:\n")
print(tfidf_df)

# Step 12: Final Observation
print("\nText preprocessing completed successfully.")