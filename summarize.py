from nltk.tokenize import sent_tokenize
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.max_length = 100000000
from collections import defaultdict
import gensim.corpora as corpora
from gensim.models.ldamulticore import LdaModel
import math

#set up preprocessing pipeline
def lemmatize(doc):
    postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    doc = [token.lemma_ for token in doc if token.pos_ in postags]
    doc = u' '.join(doc)
    return nlp.make_doc(doc)

# remove stop words
def stopwords_punct(doc):
    doc = [token.text for token in doc if (not token.is_stop) and (not token.is_punct)]
    return doc

nlp.add_pipe(lemmatize)
nlp.add_pipe(stopwords_punct)


class LdaSummarizer():

    def __init__(self, text, summary_size =5, sent_len_penalty = False):
        self.text = text
        self.summary_size = 5
        self.sent_len_penalty = sent_len_penalty

        self.sentences = None
        self.speech_tokens = None
        self.speech_topics = None
        self.summary_topics = None

        self.summary = None

    def preprocess(self):
        self.sentences = [sent for sent in sent_tokenize(self.text)]
        self.speech_tokens = list(nlp.pipe(self.sentences))

    def run_lda(self):
        dictionary = corpora.Dictionary(self.speech_tokens)
        corpus = [dictionary.doc2bow(sentence) for sentence in self.speech_tokens]
        lda = LdaModel(corpus, id2word=dictionary, num_topics=round(math.sqrt(len(self.sentences))), random_state=23, passes=5,
                       per_word_topics=True, alpha='auto')

        self.speech_topics = [lda.get_document_topics(sent, minimum_probability=0.1) for sent in corpus]

    def get_top_topics(self):
        topic_counts = defaultdict(int)
        for sent in self.speech_topics:
            for topic in sent:
                topic_counts[topic[0]] += 1

        self.summary_topics = [topic[0] for topic in (sorted(topic_counts.items(), key=lambda item: item[1]))[-self.summary_size:]]

    def get_summary(self):
        summary_sent_id = []
        for topic in self.summary_topics:

            prob = 0
            topic_sent_id = None

            for i in range(len(self.speech_topics)):
                for sent_topic in self.speech_topics[i]:
                    s_topic = sent_topic[0]
                    if self.sent_len_penalty:
                        sent_prob = sent_topic[1]/len(self.speech_tokens[i])
                    else:
                        sent_prob = sent_topic[1]

                    if topic == s_topic and sent_prob >= prob:
                        prob = sent_prob
                        topic_sent_id = i

            summary_sent_id.append(topic_sent_id)
        self.summary = " ".join([self.sentences[i] for i in sorted(summary_sent_id)])

    def summarize(self):
        self.preprocess()
        self.run_lda()

        n = round((math.sqrt(len(self.sentences))))
        if n <= self.summary_size:
            self.summary_topics = [i for i in range(n)]
        else:
            self.get_top_topics()

        self.get_summary()

def summarize_pd(text, n=5, sent_len_penalty=False):
    summarizer =  LdaSummarizer(text, summary_size=n, sent_len_penalty=sent_len_penalty)
    summarizer.summarize()
    return summarizer.summary
