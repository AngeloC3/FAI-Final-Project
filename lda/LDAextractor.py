import gensim
from gensim.utils import simple_preprocess
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
import wikipedia
import re
import gensim.corpora as corpora# Create Dictionary
from pprint import pprint# number of topics
import pandas as pd

class LDA_extractor:
    def __init__(self):
        self.sentence_splitter = r'(?<=\S{3})\.(?=\s)' 

    def make_docs_from_str(self, data : str, sentences_per_doc = 20, show_num_sentences = True):
        data = re.split(self.sentence_splitter, data)
        num_sentences = len(data)
        if show_num_sentences:
            print(f"number of sentences : {num_sentences}")

        docs = []
        for i in range(0, num_sentences, sentences_per_doc):
            docs.append(".".join(data[i:i+20]))
        return docs
    
    def sent_to_words(self, sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence), deacc = True))

    def remove_stopwords(self, texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]
    
    def topics_to_tags(self, topics):
        num_topics = len(topics)

        topics_str = topics[0][1]
        for topic in topics[1:]:
            topics_str = topics_str + f' + {topic[1]}'

        topics_str = re.sub('[", ]', '', topics_str) #remove punctuation
        tags = [x.split('*') for x in topics_str.split("+")]

        tag_dict = {}
        for tag in tags:
            if tag[1] not in tag_dict.keys():
                tag_dict[tag[1]] = (0,0)

            score, count = tag_dict[tag[1]]
            score += float(tag[0])
            count += 1
            tag_dict[tag[1]] = (round(score, 3), count)

        for tag in tag_dict.keys():
            tag_vals = {}
            sum_score, count = tag_dict[tag]
            mean_score = sum_score / tag_dict[tag][1]
            tag_vals["mean score"] = round(mean_score, 3)
            tag_vals["occurances in topics"] = count
            tag_dict[tag] = tag_vals

        return tag_dict
    

    def json_to_df(self, json_data):
        tags_list = list(json_data.keys())
        mean_scores = []
        occurances_in_topics = []
        for tag in tags_list:
            mean_scores.append(json_data[tag]['mean score'])
            occurances_in_topics.append(json_data[tag]['occurances in topics'])
        df_dict = {
            "tags" : tags_list,
            "mean scores" : mean_scores,
            "occurances in topics" : occurances_in_topics
        }
        df = pd.DataFrame(df_dict)
        return df    

    def get_keywords(self, 
                     corpus_str : str,
                     num_keywords : int,
                     sentences_per_doc = 25):
        
        self.corpus = corpus_str
        self.spd = sentences_per_doc
        docs = self.make_docs_from_str(self.corpus, sentences_per_doc = self.spd, show_num_sentences = False)
        
        # print(f"number of docs : {len(docs)}")
        # print(f"number of sentences per doc : {self.spd}")

        # data = re.sub('[,\.!?]', '', data) #remove punctuation
        docs = list(map(lambda x: re.sub('[,\.!?]', '', x), docs))
        # data = data.lower() #make lowercase
        docs = list(map(lambda x: x.lower(), docs))# Print out the first rows of papers
        data_words = list(self.sent_to_words(docs)) #string to words
        data_words = self.remove_stopwords(data_words) #remove stopwords

        id2word = corpora.Dictionary(data_words)
        texts = data_words
        corpus = [id2word.doc2bow(text) for text in texts]

        num_topics = 10# Build LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics)# Print the Keyword in the 10 topics
        doc_lda = lda_model[corpus]

        topics = lda_model.print_topics()

        tags = self.topics_to_tags(topics)

        keywords = []
        for ind, key in enumerate(tags.keys()):
            if ind < num_keywords:
                keywords.append(key)

        return keywords
