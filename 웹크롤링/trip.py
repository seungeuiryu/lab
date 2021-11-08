import time
from selenium import webdriver
import pandas as pd

outputFile = open('트립어드바이저_해운대_리뷰.txt', 'w', encoding='utf-8')

driver = webdriver.Chrome("/Users/xiu0327/PycharmProjects/team_project/chromedriver")
page = ""
url = "https://www.tripadvisor.co.kr/Attraction_Review-g297884-d1458553-Reviews-"+page+"Haeundae_Beach-Busan.html"
driver.get(url)
time.sleep(3)

page=str(10)
while (True):
    if int(page) >= 300: break
    #페이지수 조절 , 60 = 5페이지까지 크롤링
    #한 페이지당 리뷰수 = 10개. 즉, 5페이지까지 크롤링이면 리뷰데이터 개수 = 5 * 10 = 50
    review_num = "1"
    for i in range(10):
        review = driver.find_elements_by_xpath('//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div['+review_num+']/span/span/div[5]/div[1]/div/span')
        sen = review[0].text
        sen = sen.replace('\n', ' ')
        #date = driver.find_elements_by_xpath('//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div['+review_num+']/span/span/div[7]/div[1]')
        #date = driver.find_elements_by_xpath('//*[@id="tab-data-qa-reviews-0"]/div/div[5]/div[2]/span/span/div[7]/div[1]')
        #date2 = date[0].text
        #date3 = date2.replace('작성', '')
        #token ="년월"
        #for word in token:
            #date3=date3.replace(word, '-')
        #date3=date3.replace(' ', '')
        #date3=date3.replace('일', '')
        print('리뷰 : {}'.format(sen))
        outputFile.write(sen+'\n')
        #print('날짜 : {}'.format(date3))
        review_num = str(int(review_num)+1)
    url = 'https://www.tripadvisor.co.kr/Attraction_Review-g297884-d1458553-Reviews-or'+page+'-Haeundae_Beach-Busan.html'
    page = str(int(page) + 10)
    driver.get(url)

