import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import os
import re
from datetime import datetime

date = str(datetime.now())
#현재시각 문자열로 저장

date = date[:date.rfind(":")].replace(' ','_')
#date[:date.rfind(":")]는 오른쪽에서부터 ':'문자를 찾아서 찾은 index 이전의 문자열을 모두 출력한다.

date = date.replace(':','시')+'분'
print(date)

#위 코드를 작성해주는 이유 : 크롤링된 파일을 엑셀 또는 csv파일로 저장할 때 중복되는 이름이 발생하는 일이 없도록 하기 위해.

query = input('뉴스 검색 키워드 입력: ')
query = query.replace(' ', '+')
#키워드 여러개를 입렸했을 때, 공백을 +로 바꿔줌. 그러면 and연산으로 검색이 된다.
#ex) <청소년 자살>을 query에 입력했을 때 청소년+자살로 쿼리문이 바뀌면서 청소년과 자살 두 단어 모두 포함된 기사를 검색한다.

news_num = int(input('크롤링 데이터 개수 입력 : '))

since = "2021.07.24"
until = "2021.07.26"

news_url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}'+'&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds='+since+'&de='+until+'&docid=&related=0&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so%3Ar%2Cp%3Afrom20210724to20210726&is_sug_officeid=0'
req = requests.get(news_url.format(query))
#url에 뉴스 키워드 넣어서 get 요청. 키워드는 {}안에 들어간다.
soup = BeautifulSoup(req.text, 'html.parser')
#print(soup.prettify())
#추출된 페이지 소스는 print(soup.prettify())를 통해 확인할 수 있다.

news_dict = {}
idx = 0
#뉴스 개수를 세기 위한 변수

cur_page = 1
#크롤링 데이터 양이 뉴스 한 페이지의 개수보다 많으면 다음 페이지로 넘어가기 위함.

print('크롤링 시작')

while idx < news_num:
    table = soup.find('ul', {'class' : 'list_news'})
    print('테이블 출력')
    print(table)
    #ul태그 중 class 이름이 list_news인 것을 찾겠다는 뜻.
    li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})
    #re.compile('sp_nws.*')의 뜻은 현재 웹페이지 sp_nws+숫자 형태로 이름이 이루어져있기 때문에 sp_nws로 된 모든 것을 추출하여 배열로 저장함

    area_list = [li.find('div', {'class':'news_area'}) for li in li_list]
    test = soup.find_all('a', {'class':'info'})
    test2 = [k for k in test if k.text == "네이버뉴스"]
    #a_list = [area.find('a', {'class': 'news_tit'}) for area in area_list]
    #print(a_list)
    a_list = test2

    #print(a_list)

    temp_url = []

    for n in a_list[:min(len(a_list), news_num-idx)]:
        news_dict[idx] = {'title': n.get('title'),
                          'url': n.get('href')}
        #temp_url.append(n.get('href'))
        idx += 1

    cur_page += 1

    pages = soup.find('div', {'class':'sc_page_inner'})
    next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')

    print(next_page_url)
    req = requests.get('https://search.naver.com/search.naver' + next_page_url)
    soup = BeautifulSoup(req.text, 'html.parser')

news_df = pd.DataFrame(news_dict).T
news_df.to_csv('/Users/xiu0327/lab/2021_07_17/step1_crawling/result/'+date+'_news.csv')


