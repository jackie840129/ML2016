from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import numpy as np
import string
# import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from time import time
import csv
import sys

directory = str(sys.argv[1])
output_name = str(sys.argv[2])
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english",ignore_stopwords=True)

file = open(directory+'/title_StackOverflow.txt','r')
corpus = []
stop = set(stopwords.words('english'))
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

for line in file:
    line = line.strip('\n').strip(' ').lower()
    line.translate( dict.fromkeys(string.punctuation))
    line = tokenizer.tokenize(line)
    line = stem_tokens(line,stemmer)
    line = [i for i in line if i not in stop]
    line = ' '.join(line)
    corpus.append(line)
file.close()



vectorizer = TfidfVectorizer(max_df=0.5,min_df=2,max_features = None ,stop_words='english')
X = vectorizer.fit_transform(corpus)
print(X.shape)

svd = TruncatedSVD(n_components=20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)


Y = lsa.fit_transform(X)

print(Y.shape)
t0 = time()
km = KMeans(n_clusters=80, init='k-means++', max_iter=100, n_init=100,
                verbose=0)
km.fit(Y)

print('%.2f seconds'%(time()-t0))

answer = km.labels_

firstrow = True
file = open(directory+'/check_index.csv','r')
file2 = open(output_name,'w+')
file2.write("ID,Ans\n")
for row in csv.reader(file):
    if firstrow == True:
        firstrow = False
    else:
        file2.write(row[0])
        file2.write(',')
        one = int(row[1])
        two = int(row[2])
        if answer[one] == answer[two]:
            file2.write(str(1))
        else:
            file2.write(str(0))
        file2.write('\n')

        


