import wikipedia

from LDAextractor import LDA_extractor


def get_wiki_page(article_name, print_search = True):
  articles = wikipedia.search(article_name, results = 5, suggestion = True)
  print(f"articles obtained : {articles}")
  target_article = articles[0][0]
  article_page = wikipedia.page(target_article)
  article_content = {
      'title' : article_page.title,
      'url' : article_page.url,
      'content' : article_page.content,
      'images' : article_page.images,
      'links' : article_page.links
  }
  return article_content


if __name__ == '__main__':   
  lda = LDA_extractor()

  ###########     implementation parameters     ############

  article_name = "GeForce 10 series"
  sentences_per_doc = 20

  ##########################################################


  wiki_page = get_wiki_page(article_name, print_search = True)
  data = wiki_page["content"]

  keywords = lda.get_keywords(corpus_str = data, sentences_per_doc = 20)
  print(lda.json_to_df(keywords).head(-1))
  