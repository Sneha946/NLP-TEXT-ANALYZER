import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# Download data (only required the first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Read the input text
text = open("sample_input.txt", "r").read()

# Tokenize text
tokens = word_tokenize(text)

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english'))
filtered = [w.lower() for w in tokens if w.lower() not in stop_words and w.isalpha()]

# Stemming
ps = PorterStemmer()
stemmed = [ps.stem(word) for word in filtered]

# Word frequency
freq = Counter(stemmed)

print("Top Words:")
for word, count in freq.most_common(10):
    print(f"{word}: {count}")
