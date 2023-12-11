import gensim.downloader


def download_and_save_fasttext_wikipedia_model(model_name, save_path):
    # Download the FastText model trained on Wikipedia
    print("downloading...")
    model = gensim.downloader.load(model_name)

    # Save the FastText model in a Gensim-compatible format
    print("saving...")
    model.save(save_path)


model_name = "fasttext-wiki-news-subwords-300"
save_path = "fasttext_wiki_model"
download_and_save_fasttext_wikipedia_model(model_name, save_path)