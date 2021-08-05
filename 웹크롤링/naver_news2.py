import csv
import time
from selenium import webdriver
import pandas as pd
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome(executable_path="C:/Users/xiu03/lab/chromedriver.exe")

data = pd.read_csv("C:/Users/xiu03/lab/donga/data/2021-07-18_02시59분_news.csv", encoding='cp949')
url_list = data['url']

news_contents = []
for url in url_list:

    driver.get(url)
    time.sleep(3)

    xpath = "//*[@id=\"articleBodyContents\"]"
    #print(driver.find_element_by_xpath(xpath).text)
    try:
        temp_str = driver.find_element_by_xpath(xpath).text
        temp_str = temp_str.replace('\n', " ")
        news_contents.append(temp_str)
        #btnCmt = driver.find_element_by_xpath(xpath)
        print('확인')
        time.sleep(3)

    except:
        print("경로가 없습니다. continue 실행합니다.")
        continue

f = open("news_contents.txt",'w',encoding="utf-8")

df = pd.DataFrame.from_records(news_contents)
df.to_excel('lda_sample.xlsx')


for text in news_contents:
    f.writelines(text+'\n')