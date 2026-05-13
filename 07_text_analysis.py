# Step 1: Import Libraries
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Step 2: Download Required Resources
nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')

# Step 3: Create Sample Document
text = """
Data Analytics is one of the most important technologies in modern computer science.
It is used in machine learning, artificial intelligence, business analysis,
data visualization and prediction systems.
"""

# Step 4: Display Original Text
print("Original Text:\n")
print(text)

# Step 5: Tokenization
tokens = word_tokenize(text)

print("\nTokens:\n")
print(tokens)

# Step 6: POS Tagging
pos_tags = nltk.pos_tag(tokens)

print("\nPOS Tags:\n")
print(pos_tags)

# Step 7: Remove Stopwords
stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:

    if word.lower() not in stop_words:

        filtered_words.append(word)

print("\nAfter Stopword Removal:\n")
print(filtered_words)

# Step 8: Stemming
stemmer = PorterStemmer()

stemmed_words = []

for word in filtered_words:

    stemmed_words.append(stemmer.stem(word))

print("\nStemmed Words:\n")
print(stemmed_words)

# Step 9: Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in filtered_words:

    lemmatized_words.append(lemmatizer.lemmatize(word))

print("\nLemmatized Words:\n")
print(lemmatized_words)

# Step 10: Join Words Again
processed_text = " ".join(lemmatized_words)

print("\nProcessed Text:\n")
print(processed_text)

# Step 11: TF-IDF Vectorization
vectorizer = TfidfVectorizer()

tfidf = vectorizer.fit_transform([processed_text])

# Step 12: Display Feature Names
print("\nFeature Names:\n")
print(vectorizer.get_feature_names_out())

# Step 13: Display TF-IDF Matrix
print("\nTF-IDF Matrix:\n")
print(tfidf.toarray())

# Step 14: Create TF-IDF DataFrame
tfidf_df = pd.DataFrame(
    tfidf.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("\nTF-IDF DataFrame:\n")
print(tfidf_df)

# Step 15: Final Observation
print("\nText preprocessing completed successfully.")