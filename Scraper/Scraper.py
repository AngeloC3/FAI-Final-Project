from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
from selenium.webdriver.common.by import By


class Scraper:

    def scrape(self, keyword):
        '''Input : Keyword: Scaper will scarpe keyword's wikipidia page 
        Enter to close up the browser
        returns a string of the extracted info'''
        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument('window-size=1200x7000')

        string = ''

        driver = webdriver.Chrome(options=options)

        url = 'https://en.wikipedia.org/wiki/' + keyword
        driver.get(url)

        texts = driver.find_elements(By.TAG_NAME, 'p')
        for text in texts:
            para = text.text
            para = re.sub("[\(\[].*?[\)\]]", "", para)
            string = string + para
            # print(para)

        input("Press Enter to close the browser...")

        # Close the browser window
        driver.quit()
        return string
