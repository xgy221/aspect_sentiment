from nltk.corpus import stopwords as pw

stop_words = set(pw.words('english'))

s_w = ['I','am','going','to','the','mid','town','location','next']

for w in s_w:
    if w not in stop_words:
        print(w)
