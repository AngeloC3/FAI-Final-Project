import os
from fuzzywuzzy import fuzz
from keyword_extractors import *
from keybert import KeyBERT
from tqdm import tqdm
import tensorflow_hub
import gensim.downloader as genism_api

# Define your extract_keywords_max_sum and extract_keywords_mmr functions

# Specify the paths to your dataset folders
doc_folder = "SemEval2017/docsutf8"
key_folder = "SemEval2017/keys"

default_model = KeyBERT()
base_transformer_model = KeyBERT(model="all-mpnet-base-v2")
# load tensorflow
print("loading tensorflow")
embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
tensorflow_model = KeyBERT(model=embedding_model)
print("loaded tensorflow")
# load genism
print("loading genism")
ft = genism_api.load('fasttext-wiki-news-subwords-300')
genism_model = KeyBERT(model=ft)
print("loaded genism")

# model list
models = [
    (default_model, "Default Model"),
    (base_transformer_model, "Base Transformer Model"),
    (tensorflow_model, "TensorFlow Model"),
    (genism_model, "Genism Model"),
]

methods = []

# setting up models to test
for model, model_name in models:
    methods.append((no_params, model, f"Base ({model_name})", []))
    methods.append((max_sum_high, model, f"Max Sum High-Candidates ({model_name})", []))
    methods.append((max_sum_mid, model, f"Max Sum Mid-Candidates ({model_name})", []))
    methods.append((max_sum_low, model, f"Max Sum Low-Candidates ({model_name})", []))
    methods.append((max_sum_mega_small, model, f"Max Sum MegaSmall-Candidates ({model_name})", []))
    methods.append((mmr_high, model, f"MMR High Diversity ({model_name})", []))
    methods.append((mmr_mid, model, f"MMR Mid Diversity ({model_name})", []))
    methods.append((mmr_low, model, f"MMR Low Diversity ({model_name})", []))

# Iterate over the document files
doc_filenames = os.listdir(doc_folder)
for doc_filename in tqdm(doc_filenames, desc="Processing Documents"):
    doc_id = doc_filename.split(".")[0]

    # Load the document
    with open(os.path.join(doc_folder, doc_filename), "r", encoding="utf-8") as doc_file:
        doc_text = doc_file.read()

    # Load the corresponding ground truth keywords from the keyphrase file
    key_filename = doc_id + ".key"
    with open(os.path.join(key_folder, key_filename), "r", encoding="utf-8") as key_file:
        ground_truth_keywords = [line.strip() for line in key_file.readlines()]

    for method in methods:
        actual_keywords = [elem[0] for elem in method[0](doc_text, method[1])]
        scores = []
        for actual in actual_keywords:
            best = 0
            # for each actual keyword created, get its best match to one of the actual keywords and append it to scores
            for expected in ground_truth_keywords:
                best = max(best, fuzz.ratio(actual, expected))
            scores.append(best)
        avg_score = sum(scores) / len(scores)
        method[3].append(avg_score) # record the average score

output_filename = "averages.txt"

with open(output_filename, 'w') as file:
    # Sort the methods list based on avg_of_averages in descending order and write them
    sorted_methods = sorted(methods, key=lambda method: -sum(method[3]) / len(method[3]))

    for method in sorted_methods:
        avg_of_averages = sum(method[3]) / len(method[3])
        line = method[2] + ": " + str(avg_of_averages) + "\n"
        file.write(line)
