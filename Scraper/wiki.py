from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re
from selenium.webdriver.common.by import By

keyword = 'Badminton'
options = webdriver.ChromeOptions()
# options.add_argument('--ignore-certificate-errors')
options.add_experimental_option('excludeSwitches', ['enable-logging'])
# options.add_argument('headless')
options.add_argument('window-size=1200x7000')

# driver = webdriver.Chrome()
# executable_path='C:/webdrivers/chromedriver.exe', options=options
# driver.get('https://www.google.com/')
# https://en.wikipedia.org/wiki/Bitcoin

driver = webdriver.Chrome(options=options)

url = 'https://en.wikipedia.org/wiki/' + keyword
print(url)
driver.get(url)


texts = driver.find_elements(By.TAG_NAME, 'p')
for text in texts:
    para = text.text
    para = re.sub("[\(\[].*?[\)\]]", "", para)
    print(para)

input("Press Enter to close the browser...")

# Close the browser window
driver.quit()


# References:
# 1. https://medium.com/hackerdawn/scraping-from-wikipedia-using-python-and-selenium-3d64af60975d
