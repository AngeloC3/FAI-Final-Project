# Keyword Generation Project Report

This is a project that can extract keyword from a wikipedia article using different extractor and optionally match them against a given list of words to see which are most similar.


## Getting Started

- In order to run this program, you will need to:
  1. install all relevant python packages that that is required when attempting to run.
  2. Go to the Similarity module and run fast_text_downloader.py. This will download and save the fast text model as a file which is used in the program.

## Running

The main program is called keywordProgram.py. When running the program, you will encounter a few decision points:
    1. You will enter the title of a wikipedia article you wish to be scraped
    2. You will choose if you wish to see the keywords or match the article to a list of given keywords
          - if the latter is chosen, you will enter keywords that you wish to match against
    3. You will choose whether you want to use the KeyBERT extractor or the LDA extractor for keyword extraction
    4. If you chose to match the article to a keyword list, you will choose how many of the top similar keyword you wish to see

