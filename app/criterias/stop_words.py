from nltk.corpus import stopwords

def get_stop_words(lan):
        '''
        get stopwords for a language
        '''
        stopWords = set(stopwords.words(lan))
        return stopWords
