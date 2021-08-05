#필요한 라이브러리들
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import re

#url

def naver_movie():
    code = ['184318']
    #영화코드. 184318 = 블랙위도우
    for code_num in code:
        BASE_URL = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code='+code_num+'&amp;type=after&amp;onlyActualPointYn=N&amp;onlySpoilerPointYn=N&amp;order=newest&amp;page=2&page='
        page_num = 1
        id = 'pagerTagAnchor'
        #각 페이지별 고유 id

        driver = webdriver.Chrome('C:/Users/xiu03/lab/chromedriver.exe')
        star_score=[] #별점이 들어갈 List
        content = [] #댓글이 들어갈 List
        while (page_num < 100): #무한루프 #try except에 넣은 이유는 더이상 페이지가 없으면 에러를 띄워 멈추도록 하기위해서임
            try:
                URL = BASE_URL+str(page_num) #url과 page_num 을 통해 각 페이지의 url 을 만들어줌
                page_id = id + str(page_num)
                driver.get(URL) #크롬드라이버 사용
                driver.find_element(By.ID,page_id)#find가 실패하면 페이지가 없는것이므로 error 발생 break

                for num in range(10): #각페이지에 10개의 댓글이있음
                    id_name='_filtered_ment_'+str(num) #id 의 id값
                    comment = driver.find_element_by_id(id_name).text #댓글부분 추출
                    comment = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','',comment) #한글을 제외하고 삭제
                    if comment != '': #삭제를 했을때 공백이 아니라면 list 에 추가
                        content.append(comment)
                        star_score.append(int(driver.find_element_by_css_selector('body > div > div > div.score_result > ul > li:nth-child('+str(num+1)+') > div.star_score > em').text))
                        #별점도 list 에 추가
                page_num+=1 #페이지 번호 1 상승
                print(page_num)
            except:
                break
        driver.quit()

        def star_score_eval(star_score): #별점을 통해 긍정부정 판단해주기
            change_star = []
            for star in star_score:
                if star >= 7: #7점 이상인경우
                    change_star.append(1)
                else: #7점 미만인경우
                    change_star.append(0)

            return change_star

        change_star = star_score_eval(star_score)
        replys = list(zip(star_score,content,change_star)) #pandas 를 사용해서 csv 파일로 만들어주기
        col = ['star_score','document','label']
        data_frame = pd.DataFrame(replys,columns=col)
        full_name = str(code_num) +'_data.csv'
        data_frame.to_csv(full_name,sep=',',header=True)
        print('work it')

naver_movie()
print('완료')