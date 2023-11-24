from similarity.Similarity import Similarity
from lda.LDAextractor import LDA_extractor
from KeyBert.keybert_extractor import KeyBertExtractor
from Scraper.Scraper import Scraper

print("Welcome to the keyword generator.\n")
topic = input(
    "Please enter the wiki topic for which you want to generate keywords: ")
mode = bool(int(input(
    "Enter 0 if you want to use see the keyword itself and 1 if you want to match it with a list of keywords: ")))
list_of_topics = []
if mode:
    while True:

        val = input(
            "Press *q to quit. Enter any other keyword to include it in the list: ")
        if val == "*q":
            break
        else:
            list_of_topics.append(val)
else:
    list_of_topics.append(topic)


# Scrape the webpage
scraper = Scraper()
document = scraper.scrape(topic)

e_type = bool(int(input("Enter 0 for keyBERT extractor and 1 for LDA extractor: ")))


if not e_type:  # Use keyBERT
    e = KeyBertExtractor()
    keyword = e.extract_keywords(document)
else:
    l = LDA_extractor()
    keyword = l.get_keywords(document, 10)

if mode:
    n = int(input("How many of the top n similar keywords would you like to see? "))
    s = Similarity(n)
    max_sim =[]
    d = dict()
    for i in keyword:
        keys = s.top_n_similar_words(list_of_topics, i)
        for j in keys:
            if j in d.keys():
                d[j]+=1
            else:
                d[j]=1
    for i in range(n):
        k = max(d, key=lambda k: d[k])
        max_sim.append(k)
        d[k] = 0

    print(max_sim)
else:
    print(keyword)
