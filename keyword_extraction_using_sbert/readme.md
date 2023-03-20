## (컴퓨터 & 데이터 분야) 도서 키워드 추출 모델 구현

<br/>

> 키워드 추출관련 메서드는 key_extraction.py에 구현 되어있습니다.

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

    Some weights of the model checkpoint at monologg/koelectra-base-v3-discriminator were not used when initializing ElectraModel: ['discriminator_predictions.dense_prediction.weight', 'discriminator_predictions.dense_prediction.bias', 'discriminator_predictions.dense.bias', 'discriminator_predictions.dense.weight']
    - This IS expected if you are initializing ElectraModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing ElectraModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


    도서 데이터 수 :  5895

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
      <td>9791192932057</td>
      <td>챗GPT</td>
      <td>[AI는 이미 당신보다 똑똑하다, 너무 똑똑한 AI의 출현 위기인가 기회인가, 고도...</td>
      <td>[출시된 지 얼마 되지도 않아 세상을 뒤흔든 챗GPT는 지금까지 나온 모든 인공지능...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9788931467604</td>
      <td>오피스 초보 직장인을 위한 엑셀&amp;파워포인트&amp;워드&amp;윈도우 11</td>
      <td>[Chapter 워크시트 관리 기술, 엑셀의 시작 화면과 화면 구성 살펴보기, 자주...</td>
      <td>[모든 버전 에서 사용 가능한 오피스 통합 도서이다, 오피스 까막눈을 위한 가지 오...</td>
      <td>[PART 엑셀 CHAPTER l 워크시트 관리 기술엑셀은 많은 양의 데이터를 분석...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9791140702787</td>
      <td>쉽게 시작하는 쿠버네티스</td>
      <td>[장 쿠버네티스의 등장, 컨테이너 환경으로의 진화, 쿠버네티스를 학습하기 전에 알아...</td>
      <td>[]</td>
      <td>[모든 것은 기본에서 시작한다 가볍지만 알차게 배우는 쿠버네티스 쿠버네티스는 컨테이...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9791140703210</td>
      <td>세상에서 가장 쉬운 코딩책</td>
      <td>[프롤로그, 이 책의 특징, 학습 로드맵, 코딩 두뇌 패치하기, 기초편 코딩의 기초...</td>
      <td>[우리는 디지털과는 떼려야 뗄 수 없는 일상을 살고 있다, 스마트폰으로 업무를 하고...</td>
      <td>[내용 소개디자이너에서 개발자로 커리어전환한 저자의 코딩 공부와 개발자 취직의 A ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9791140703029</td>
      <td>소프트웨어 코딩 대회를 위한 파이썬 문제 풀이 100</td>
      <td>[문제 글자 출력하기, 문제 숫자 저장하기, 문제 makit 곱하기, 문제 하나 빼...</td>
      <td>[가지 문제를 풀면서 배우는 파이썬 프로그래밍 기초, 이 책은 아주 간단한 문제부터...</td>
      <td>[가지 문제를 풀면서 배우는 파이썬 프로그래밍 기초 코딩 대회 입문자를 위한 맞춤형...</td>
    </tr>
  </tbody>
</table>
</div>

## 키워드 추출 예시

```python
# extract keywords
docs_keywords = key.extract_keyword(scraping_result.iloc[[4294]])

# result
result = pd.DataFrame(docs_keywords)

print('키워드 추출 예시\n')
print('도서제목 : ', scraping_result.iloc[4294].title)
pd.DataFrame(result.keywords.values[0])
```

    키워드 추출 예시

    도서제목 :  파이썬 라이브러리로 배우는 딥러닝 입문과 응용

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
      <td>오토인코더</td>
    </tr>
    <tr>
      <th>1</th>
      <td>머신러닝</td>
    </tr>
    <tr>
      <th>2</th>
      <td>뉴럴</td>
    </tr>
    <tr>
      <th>3</th>
      <td>컨볼루션</td>
    </tr>
    <tr>
      <th>4</th>
      <td>딥러닝</td>
    </tr>
    <tr>
      <th>5</th>
      <td>알고리즘</td>
    </tr>
    <tr>
      <th>6</th>
      <td>파이썬</td>
    </tr>
    <tr>
      <th>7</th>
      <td>볼츠</td>
    </tr>
    <tr>
      <th>8</th>
      <td>소스</td>
    </tr>
    <tr>
      <th>9</th>
      <td>라이브러리</td>
    </tr>
    <tr>
      <th>10</th>
      <td>기울기</td>
    </tr>
    <tr>
      <th>11</th>
      <td>하기</td>
    </tr>
    <tr>
      <th>12</th>
      <td>케라스</td>
    </tr>
    <tr>
      <th>13</th>
      <td>요약</td>
    </tr>
    <tr>
      <th>14</th>
      <td>게임</td>
    </tr>
    <tr>
      <th>15</th>
      <td>함수</td>
    </tr>
    <tr>
      <th>16</th>
      <td>제한</td>
    </tr>
    <tr>
      <th>17</th>
      <td>인공지능</td>
    </tr>
    <tr>
      <th>18</th>
      <td>예시</td>
    </tr>
    <tr>
      <th>19</th>
      <td>데이터</td>
    </tr>
  </tbody>
</table>
</div>

## 키워드 추출 상세

### 1. 데이터 전처리

```python
min_count = 3
min_length = 2
doc = scraping_result.iloc[4294]

print(f'도서 정보 \n \n {doc} \n \n')


raw_data = key._convert_series_to_list(doc)
print(f'1. 도서 정보를 list로 통합 -> {len(raw_data)} 개 단어')
print(f'\n \n {raw_data[:10]}.... \n \n')

keyword_list = key._extract_keywords(raw_data)
print(f'2. 형태소 분석기를 활용해 명사만을 추출 -> {len(keyword_list)} 개 단어')
print(f'\n \n {keyword_list[:10]}.... \n \n')

translated_keyword_list = key._map_english_to_korean(keyword_list)
print(f'3. 영단어를 한글로 변환(ex python -> 파이썬) -> {len(translated_keyword_list)} 개 단어')
print(f'\n \n {translated_keyword_list[:10]}.... \n \n')

refined_keyword_list = key._eliminate_min_count_words(translated_keyword_list, min_count)
print(f'4. 최소 3번이상 반복 사용되는 단어만 추출 -> {len(refined_keyword_list)} 개 단어')
print(f'\n \n {refined_keyword_list[:10]}.... \n \n')

result = list(filter(lambda x: len(x) >= min_length, refined_keyword_list))
print(f'5. 단어 길이가 최소 한개 이상인 단어만 추출 -> {len(result)} 개 단어')
print(f'\n \n {result[:10]}.... \n \n')
```

    도서 정보

     isbn13                                           9791188621354
    title                                파이썬 라이브러리로 배우는 딥러닝 입문과 응용
    toc          [머신러닝이란, 다양한 머신러닝 접근법, 지도학습, 비지도학습, 강화학습, 머신러닝...
    intro                                                       []
    publisher    [컴퓨터 비전 인공지능 음성 및 데이터 분석을 위한 차세대 핵심 테크닉 실제 적용 ...
    Name: 4294, dtype: object


    1. 도서 정보를 list로 통합 -> 610 개 단어


     ['파이썬', '라이브러리로', '배우는', '딥러닝', '입문과', '응용', '머신러닝이란', '다양한', '머신러닝', '접근법']....


    2. 형태소 분석기를 활용해 명사만을 추출 -> 547 개 단어


     ['파이썬', '라이브러리', '딥러닝', '입문', '응용', '머신러닝', '다양', '머신러닝', '접근법', '지도']....


    3. 영단어를 한글로 변환(ex python -> 파이썬) -> 547 개 단어


     ['파이썬', '라이브러리', '딥러닝', '입문', '응용', '머신러닝', '다양', '머신러닝', '접근법', '지도']....


    4. 최소 3번이상 반복 사용되는 단어만 추출 -> 55 개 단어


     ['파이썬', '라이브러리', '딥러닝', '머신러닝', '다양', '학습', '기법', '알고리즘', '실생활', '적용']....


    5. 단어 길이가 최소 한개 이상인 단어만 추출 -> 50 개 단어


     ['파이썬', '라이브러리', '딥러닝', '머신러닝', '다양', '학습', '기법', '알고리즘', '실생활', '적용']....

### 2. 키워드 임베딩 및 문서 임베딩 생성

```python
from pprint import pprint
doc = scraping_result.iloc[4294]

print(f'-- 도서제목 -- \n {doc.title} \n \n')

keyword_list = key.extract_keyword_list(doc)
print(f'도서에 대한 키워드 후보 : {len(result)} 개 단어')
print(f'{result[:10]}.... \n \n')


keyword_embedding = key.create_keyword_embedding(doc)
doc_embedding = key.create_doc_embedding(doc)
```

    -- 도서제목 --
     파이썬 라이브러리로 배우는 딥러닝 입문과 응용


    도서에 대한 키워드 후보 : 50 개 단어
    ['파이썬', '라이브러리', '딥러닝', '머신러닝', '다양', '학습', '기법', '알고리즘', '실생활', '적용']....

### 3. 코사인 유사도를 활용에 문서와 연관성 높은 키워드 추출

```python
co_sim_score =key._calc_cosine_similarity(doc_embedding, keyword_embedding).flatten()

keyword = dict(zip(keyword_list, co_sim_score))
sorted_keyword = sorted(keyword.items(), key=lambda k: k[1], reverse=True)

print(f'-- 키워드 추출 결과(20개 요약)--')
pprint(sorted_keyword[:20])
```

    -- 키워드 추출 결과(20개 요약)--
    [('오토인코더', 0.95437026),
     ('머신러닝', 0.94073564),
     ('뉴럴', 0.92509925),
     ('컨볼루션', 0.9216925),
     ('딥러닝', 0.9182806),
     ('알고리즘', 0.9138457),
     ('파이썬', 0.91044587),
     ('볼츠', 0.90905386),
     ('소스', 0.9054553),
     ('라이브러리', 0.903215),
     ('기울기', 0.8987693),
     ('하기', 0.89817643),
     ('케라스', 0.8977781),
     ('요약', 0.89715064),
     ('게임', 0.88808674),
     ('함수', 0.88727576),
     ('제한', 0.8852492),
     ('인공지능', 0.88420147),
     ('예시', 0.88184154),
     ('데이터', 0.8805901)]

## 추출한 키워드를 활용해 도서 검색 기능 구현(W2V 활용)

### 추출한 도서 키워드 목록 및 학습된 W2V 모델 불러오기

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

### 검색 서비스 구현

```python
import numpy as np

# 키워드 검색
search = ['자연어', '딥러닝']
print('사용자 검색 키워드 : ', search)
print('')

# 키워드 확장
recommand_keyword = w2v_model.most_similar(positive=search, topn=15)
np_recommand_keyword = np.array(list(map(lambda x: x[0], recommand_keyword)))
print('W2V을 활용한 키워드 확장 :', np_recommand_keyword)
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

    W2V을 활용한 키워드 확장 : ['연어' 'nlp' '머신러닝' '신경망' '인공신경망' 'bert' '파이토치' '트랜스포머' 'cnn' 'transformer'
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
