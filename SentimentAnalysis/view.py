from django.http import HttpResponse
from django.shortcuts import render
import nltk
import re
import numpy as np
import pandas as pd
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
df = pd.read_csv("spam.csv")
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ',df['Message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pd.Series(corpus),df.spam)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)


# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)


def index(request):
    params = {'name': 'Shashank', 'place': 'Mars'}
    return render(request, 'index.html',params)

def analyze(request):
    djtext=request.POST.get('text','default')
    print(djtext)
    removepunc = request.POST.get('removepunc', 'off')
    fullcaps = request.POST.get('fullcaps', 'off')
    newlineremover = request.POST.get('newlineremover', 'off')
    extraspaceremover = request.POST.get('extraspaceremover', 'off')
    textsentiment=request.POST.get('textsentiment', 'off')
    emailspam = request.POST.get('emailspam', 'off')
    wordcount= request.POST.get('wordcount','off')
    sentencecount=request.POST.get('sentencecount','off')

    # Check which checkbox is on
    if  textsentiment =="on":
        sid = SentimentIntensityAnalyzer()
        doc1 = djtext
        dctn=sid.polarity_scores(doc1)
        return render(request, 'analyze2.html', dctn)

    elif wordcount == "on":
        dt={'len':0}
        r=nltk.word_tokenize(djtext)
        dt['len']=len(r)
        return render(request,'analyze4.html',dt)

    elif sentencecount == "on":
        dt = {'len': 0}
        r = nltk.sent_tokenize(djtext)
        dt['len'] = len(r)
        return render(request, 'analyze4.html', dt)

    elif emailspam =="on":
        ed={'test':0}
        emails = [djtext
        ]

        emails_count = v.transform(emails)
        ans=list(model.predict(emails_count))

        if ans[0]==0:
            ed['test']="Ham"
        else:
            ed['test']="Spam"
    
        return render(request, 'analyze3.html', ed)


    elif removepunc == "on":
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        analyzed = ""
        for char in djtext:
            if char not in punctuations:
                analyzed = analyzed + char
        params = {'purpose': 'Removed Punctuations', 'analyzed_text': analyzed}
        return render(request, 'analyze.html', params)

    elif (fullcaps == "on"):
        analyzed = ""
        for char in djtext:
            analyzed = analyzed + char.upper()

        params = {'purpose': 'Changed to Uppercase', 'analyzed_text': analyzed}
        # Analyze the text
        return render(request, 'analyze.html', params)

    elif (extraspaceremover == "on"):
        analyzed = ""
        for index, char in enumerate(djtext):
            if not (djtext[index] == " " and djtext[index + 1] == " "):
                analyzed = analyzed + char

        params = {'purpose': 'Removed NewLines', 'analyzed_text': analyzed}
        # Analyze the text
        return render(request, 'analyze.html', params)

    elif (newlineremover == "on"):
        analyzed = ""
        for char in djtext:
            if char != "\n" and char!='\r':
                analyzed = analyzed + char

        params = {'purpose': 'Removed NewLines', 'analyzed_text': analyzed}
        # Analyze the text
        return render(request, 'analyze.html', params)
    else:
        return HttpResponse("Error")

