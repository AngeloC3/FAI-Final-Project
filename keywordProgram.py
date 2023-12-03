from similarity.Similarity import Similarity
from lda.LDAextractor import LDA_extractor
from KeyBert.keybert_extractor import KeyBertExtractor
from Scraper.Scraper import Scraper


def get_zero_one_bool(message):
    while True:
        try:
            mode = bool(int(input(message)))
            return mode
        except ValueError:
            print("You must enter 0 or 1")


def numbered_print(elems):
    print()
    for index, elem in enumerate(elems):
        print(f"{index + 1}. {elem}")


if __name__ == '__main__':   
    print("Welcome to the keyword generator.\n")
    topic = input(
        "Please enter the wiki topic for which you want to generate keywords: ")
    mode = get_zero_one_bool("Enter 0 if you want to use see the keyword itself and 1 if you want to match it with a list of keywords: ")
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
    print("Scraping...")
    scraper = Scraper()
    document = scraper.scrape(topic)

    e_type = get_zero_one_bool("Enter 0 for keyBERT extractor and 1 for LDA extractor: ")
    print("Extracting keywords...")

    if not e_type:  # Use keyBERT
        e = KeyBertExtractor()
        keyword = e.extract_keywords(document)
    else:
        l = LDA_extractor()
        keyword = l.get_keywords(corpus_str = document,
                                 num_keywords = 5,
                                 sentences_per_doc = 25)
    if mode:
        while True:
            try:
                n = int(input("How many of the top n similar keywords would you like to see? "))
                if n > len(list_of_topics):
                    print("You have chosen an n value greater than the amount of keywords given, so all keywords will be shown.")
                    n = len(list_of_topics)
                break
            except ValueError:
                print("N must be a number!")
        print("Getting similarity data...")
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

        numbered_print(max_sim)
    else:
        numbered_print(keyword)


