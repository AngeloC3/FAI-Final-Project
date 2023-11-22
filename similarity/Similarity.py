import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors

class Similarity:

    def __init__(n):
        self.n = n
    def calculate_phrase_similarity(self,phrase1, phrase2):
        word_vectors = KeyedVectors.load("fasttext_wiki_model")
        # Tokenize and get vectors for each word in the phrases
        tokens1 = phrase1.split()
        tokens2 = phrase2.split()

        # Remove words not in the vocabulary
        tokens1 = [word for word in tokens1 if word in word_vectors]
        tokens2 = [word for word in tokens2 if word in word_vectors]

        if not tokens1 or not tokens2:
            return None  # No common words in the vocabulary

        # Calculate the average vector representation for each phrase
        avg_vector1 = word_vectors[tokens1].mean(axis=0)
        avg_vector2 = word_vectors[tokens2].mean(axis=0)

        # Calculate cosine similarity between the average vectors
        similarity = 1 - cosine(avg_vector1, avg_vector2)

        return similarity

    def top_n_similar_words(self,list_of_classes,generated_keyword):
        n = self.n
        max_sim=[]
        similarities = dict()
        for i in list_of_classes:
            similarities[i] = calculate_phrase_similarity(i, generated_keyword)
        for i in range(n):
            k = max(similarities, key=lambda k: similarities[k])
            max_sim.append(k)
            similarities[k] = 0
        return max_sim

    


# phrase1 = "apple"
# phrase2 = "banana"

# phrase_similarity = calculate_phrase_similarity(phrase1, phrase2)

# if phrase_similarity is not None:
#     print(f"Similarity between '{phrase1}' and '{phrase2}': {phrase_similarity:.2f}")
# else:
#     print("No common words in the vocabulary.")

# list1 = ['apple','pear','strawberry','blueberry','basketball']
# keyword = 'banana'
# print(top_n_similar_words(list1,keyword,3))