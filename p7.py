import nltk
input = "If you're looking for random paragraphs, you've come to the right place. When a random word or a random sentence isn't quite enough, the next logical step is to find a random paragraph. We created the Random Paragraph Generator with you in mind. The process is quite simple. Choose the number of random paragraphs you'd like to see and click the button. Your chosen number of paragraphs will instantly appear."

word = nltk.word_tokenize(input)
word

sent = nltk.sent_tokenize(input)
sent

pos = nltk.pos_tag(word)
pos

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
rem = []
for words in word:
    if words not in stop:
        rem.append(words)
rem

from nltk.stem import PorterStemmer
stm = PorterStemmer()

stemm = []

for words in word:
    stemm.append(stm.stem(words))
stemm

from nltk.stem import WordNetLemmatizer

wl = WordNetLemmatizer()

lemm = []

for i in word:
    lemm.append(wl.lemmatize(i))

lemm

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
count = CountVectorizer()
tf = count.fit_transform(word)
for i1, i2 in zip(count.get_feature_names(), np.ravel(tf.sum(axis = 0))):
    print(i1, i2)

from sklearn.feature_extraction.text import TfidfVectorizer
count = TfidfVectorizer()
tf = count.fit(word)
for i1, i2 in zip(count.get_feature_names(), tf.idf_):
    print(i1, i2)