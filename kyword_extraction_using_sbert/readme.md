### 도서관 장서 데이터에 대한 키워드 추출(컴퓨터 & 데이터 분야 한정)


```python
from key_extraction import keywordExtractor
from transformers import ElectraModel, ElectraTokenizerFast
import numpy as np
import pandas as pd

# load model and tokenizer
name = "monologg/koelectra-base-v3-discriminator"
model = ElectraModel.from_pretrained(name)
tokenizer = ElectraTokenizerFast.from_pretrained(name)

# load keywordExtractor
key = keywordExtractor(model,tokenizer,dir='data/preprocess/eng_han.csv')

# load scraping_data
scraping_result = pd.read_parquet('data/scraping_result.parquet')
print('도서 데이터 수 : ', len(scraping_result))
print('')
scraping_result.head()
```

    Some weights of the model checkpoint at monologg/koelectra-base-v3-discriminator were not used when initializing ElectraModel: ['discriminator_predictions.dense_prediction.bias', 'discriminator_predictions.dense.bias', 'discriminator_predictions.dense.weight', 'discriminator_predictions.dense_prediction.weight']
    - This IS expected if you are initializing ElectraModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing ElectraModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    도서 데이터 수 :  3979
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isbn13</th>
      <th>title</th>
      <th>toc</th>
      <th>intro</th>
      <th>publisher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9791140702787</td>
      <td>쉽게 시작하는 쿠버네티스</td>
      <td>[장 쿠버네티스의 등장, 컨테이너 환경으로의 진화, 쿠버네티스를 학습하기 전에 알아...</td>
      <td>[]</td>
      <td>[모든 것은 기본에서 시작한다 가볍지만 알차게 배우는 쿠버네티스 쿠버네티스는 컨테이...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9791163034254</td>
      <td>깡샘의 안드로이드 앱 프로그래밍 with 코틀린</td>
      <td>[첫째마당 안드로이드 앱 개발 준비하기, 개발 환경 준비하기, 안드로이드 스튜디오 ...</td>
      <td>[안드로이드 코틀린 분야 위 도서였던 개정판에 이어 개정 판이 출간되었다, 이번 판...</td>
      <td>[이 책의 특징 안드로이드 티라미수을 기준으로 내용 및 소스를 업데이트했습니다, 전...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9788931467604</td>
      <td>오피스 초보 직장인을 위한 엑셀&amp;파워포인트&amp;워드&amp;윈도우 11</td>
      <td>[Chapter 워크시트 관리 기술, 엑셀의 시작 화면과 화면 구성 살펴보기, 자주...</td>
      <td>[모든 버전 에서 사용 가능한 오피스 통합 도서이다, 오피스 까막눈을 위한 가지 오...</td>
      <td>[PART 엑셀 CHAPTER l 워크시트 관리 기술엑셀은 많은 양의 데이터를 분석...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9791192932057</td>
      <td>챗GPT</td>
      <td>[AI는 이미 당신보다 똑똑하다, 너무 똑똑한 AI의 출현 위기인가 기회인가, 고도...</td>
      <td>[출시된 지 얼마 되지도 않아 세상을 뒤흔든 챗GPT는 지금까지 나온 모든 인공지능...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9788966263677</td>
      <td>CPython 파헤치기</td>
      <td>[소스 코드에 포함된 것들, 장 개발 환경 구성하기, 편집기와 통합 개발 환경, 비...</td>
      <td>[파이썬이 인터프리터 레벨에서 작동하는 방식을 이해하면 파이썬의 기능을 최대한 활용...</td>
      <td>[]</td>
    </tr>
  </tbody>
</table>
</div>



### 키워드 추출
- 키워드 추출과 관련한 매서드는 `key_extracion.py`를 참고 바랍니다.


```python
# extract keywords
docs_keywords = key.extract_keyword(scraping_result.iloc[:5])

# result
result = pd.DataFrame(docs_keywords)
print('도서제목 : ', scraping_result.iloc[4].title)
pd.DataFrame(result.iloc[4].keywords)
```

    도서제목 :  CPython 파헤치기





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>바이트코드</td>
    </tr>
    <tr>
      <th>1</th>
      <td>구조체</td>
    </tr>
    <tr>
      <th>2</th>
      <td>프로파일링</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인터프리터</td>
    </tr>
    <tr>
      <th>4</th>
      <td>컴파일</td>
    </tr>
    <tr>
      <th>5</th>
      <td>멀티</td>
    </tr>
    <tr>
      <th>6</th>
      <td>딕셔너리</td>
    </tr>
    <tr>
      <th>7</th>
      <td>api</td>
    </tr>
    <tr>
      <th>8</th>
      <td>심벌</td>
    </tr>
    <tr>
      <th>9</th>
      <td>디버거</td>
    </tr>
    <tr>
      <th>10</th>
      <td>컴파일러</td>
    </tr>
    <tr>
      <th>11</th>
      <td>가비지</td>
    </tr>
    <tr>
      <th>12</th>
      <td>파이썬</td>
    </tr>
    <tr>
      <th>13</th>
      <td>macos</td>
    </tr>
    <tr>
      <th>14</th>
      <td>벤치마크</td>
    </tr>
    <tr>
      <th>15</th>
      <td>연결</td>
    </tr>
    <tr>
      <th>16</th>
      <td>코드</td>
    </tr>
    <tr>
      <th>17</th>
      <td>소스</td>
    </tr>
    <tr>
      <th>18</th>
      <td>dtrace</td>
    </tr>
    <tr>
      <th>19</th>
      <td>내부</td>
    </tr>
  </tbody>
</table>
</div>



### 추출한 키워드를 활용해 도서 검색 기능 구현(W2V 활용)


```python
from gensim.models import keyedvectors
import pickle

def read_pkl(dir: str) -> list:
    # for reading also binary mode is important
    with open(dir, "rb") as fp:
        n_list = pickle.load(fp)
        return n_list

# load book keywords 
isbn_list,book_keyword = read_pkl('data/data_for_search')

# load trained w2v model
w2v_model = keyedvectors.load_word2vec_format('data/w2v')

pd.DataFrame([isbn_list,book_keyword]).T.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9791140702688</td>
      <td>[도전, 파이썬, 알고리즘, np, 컴퓨터, 자료, 구현, 데이터, 분할, 그래프,...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9791192469546</td>
      <td>[마이크로컨트롤러, 라즈베리파이, 마이크로, 키패드, 온도, 파이썬, 피코, spi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9791169210140</td>
      <td>[신경망, 정규화, 로지스틱, 텐서, sgd, 딥러닝, 심층, 기초, mnist, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9791140702121</td>
      <td>[httpmessageconverter, xml, properties, 메서드, r...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9791191905236</td>
      <td>[밥값, 경험, 프로덕트, 커뮤니티, 중요, 공유, 준비, 일치, cto, 마이너,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np  

# 키워드 검색
search = ['자연어', '딥러닝']
print('사용자 검색 키워드 : ', search)
print('')

# 키워드 확장 
recommand_keyword = w2v_model.most_similar(positive=search, topn=15)
np_recommand_keyword = np.array(list(map(lambda x: x[0], recommand_keyword)))
print('도서 검색을 위한 키워드 확장 :', np_recommand_keyword)
print('')

# 키워드와 유사한 도서 검색 
user_point = np.isin(book_keyword, np.array(search)).sum(axis=1)
recommand_point = np.isin(book_keyword, np_recommand_keyword).sum(axis=1)

total_point = (user_point * 3) + recommand_point
top_k_idx = np.argsort(total_point)[::-1][:20]

# Isbn 및 연관 점수 저장
result  = dict(zip(isbn_list[top_k_idx], total_point[top_k_idx]))

# 도서정보 추출
book_info = pd.read_parquet('data/book_info.parquet')
BM = book_info.isbn13.isin(list(result.keys()))
books_recommandation_result = book_info[["bookname", "isbn13"]][BM].sort_values(
    by="isbn13", key=lambda x: x.map(result), ascending=False
).reset_index(drop=True).drop(columns='isbn13')

print('키워드에 따른 상위 20개 도서 추천 결과')
books_recommandation_result


```

    사용자 검색 키워드 :  ['자연어', '딥러닝']
    
    도서 검색을 위한 키워드 확장 : ['연어' 'nlp' '머신러닝' '신경망' '인공신경망' 'bert' '파이토치' '트랜스포머' 'cnn' 'transformer'
     '밑바닥' 'rnn' '러닝' 'lenet' 'ann']
    
    키워드에 따른 상위 20개 도서 추천 결과





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bookname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Do it! BERT와 GPT로 배우는 자연어 처리 - 트랜스포머 핵심 원리와 허깅...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>파이썬 텍스트 마이닝 완벽 가이드</td>
    </tr>
    <tr>
      <th>2</th>
      <td>딥러닝 파이토치 교과서 : 기초부터 CNN, RNN, 시계열 분석, 성능 최적화, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>케라스 2.x 프로젝트 :실전 딥러닝 모델 구축과 훈련 9가지 프로젝트</td>
    </tr>
    <tr>
      <th>4</th>
      <td>딥러닝에 목마른 사람들을 위한 PyTorch</td>
    </tr>
    <tr>
      <th>5</th>
      <td>코딩은 처음이라 with 딥러닝 :캐글 &amp; 케라스로 시작하는 딥러닝 모델 다루기</td>
    </tr>
    <tr>
      <th>6</th>
      <td>파이토치로 배우는 자연어 처리</td>
    </tr>
    <tr>
      <th>7</th>
      <td>파이썬 딥러닝 파이토치</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(한 권으로 다지는) 머신러닝&amp;딥러닝 with 파이썬 :인공지능 핵심 개념과 사용 ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>텐서플로 2로 배우는 금융 머신러닝 :텐서플로와 Scikit-learn으로 금융 경...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>인공지능을 위한 수학 :꼭 필요한 것만 골라 배우는 인공지능 맞춤 수학</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AI 상식사전</td>
    </tr>
    <tr>
      <th>12</th>
      <td>케라스 창시자에게 배우는 딥러닝 - 창시자의 철학까지 담은 머신 러닝/딥러닝 핵심 ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(기초부터 배우는) 인공지능</td>
    </tr>
    <tr>
      <th>14</th>
      <td>자연어 처리와 딥러닝 - 딥러닝으로 바라보는 언어에 대한 생각, 2021년 대한민국...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>핸즈온 머신러닝.딥러닝 알고리즘 트레이딩 :파이썬, Pandas, NumPy, Sc...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(놀랍게 쉬운) 인공지능의 이해와 실습 :블록 프로그래밍 실습</td>
    </tr>
    <tr>
      <th>17</th>
      <td>만들면서 배우는 파이토치 딥러닝</td>
    </tr>
    <tr>
      <th>18</th>
      <td>핸즈온 머신러닝 :사이킷런, 케라스, 텐서플로 2를 활용한 머신러닝, 딥러닝 완벽 실무</td>
    </tr>
    <tr>
      <th>19</th>
      <td>처음 배우는 인공지능 :개발자를 위한 인공지능 알고리즘과 인프라 기초</td>
    </tr>
  </tbody>
</table>
</div>


