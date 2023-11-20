from sentence_transformers import SentenceTransformer


def no_params(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english')


def max_sum_high(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_maxsum=True, nr_candidates=60, top_n=5)


def max_sum_mid(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_maxsum=True, nr_candidates=40, top_n=5)


def max_sum_low(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_maxsum=True, nr_candidates=20, top_n=5)


def max_sum_mega_small(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_maxsum=True, nr_candidates=5, top_n=5)


def mmr_high(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_mmr=True, diversity=0.75)


def mmr_mid(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_mmr=True, diversity=0.5)


def mmr_low(doc, kw_model):
    return kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english',
                                     use_mmr=True, diversity=0.25)
