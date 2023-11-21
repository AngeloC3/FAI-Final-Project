from keybert import KeyBERT


class KeyBertExtractor:
    def __init__(self):
        self.model = KeyBERT(model="all-mpnet-base-v2")

    def extract_keywords(self, doc, ngram_range=(1, 3), use_mmr=False, diversity=.25):
        if use_mmr:
            return self.model.extract_keywords(doc, keyphrase_ngram_range=ngram_range, stop_words='english',
                                               use_mmr=True, diversity=diversity)
        return self.model.extract_keywords(doc, keyphrase_ngram_range=ngram_range, stop_words='english')
