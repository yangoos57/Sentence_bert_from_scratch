from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Hannanum
import pandas as pd
import ast
import re

"""
electra를 활용해 sentence embedding, Doc embedding을 수행하기 위한 여러 전처리 함수를 정리하였음

"""


def merge_series_to_str(series: pd.Series) -> str:

    """
    pd.Series 데이터를 하나의 str으로 통합하는 함수

    """
    if isinstance(series, pd.Series):
        val_array = series.values
    else:
        assert type(series) == list
        val_array: list = series

    lst = []
    for item in val_array:
        if item[0] == "[":
            """
            str으로 저장된 list 자료형을 다시 list로 변환하는 함수
            ** list type을 csv 저장 시 str 타입으로 저장됨.
            """
            item = ast.literal_eval(item)
            lst.extend(item)
        else:
            lst.append(item)

    # 리스트 내 ''제거
    lst = list(filter(None, lst))

    print("변환한 도서정보 : ", lst[0])

    return re.sub(r"[^\w\s]", "", " ".join(lst))


def trans_eng_to_han(words: str, englist, print_on=False) -> list:
    """
    preprocess/englist.csv 내 한,영 문자를 활용해
    영문 용어를 한글로 바꾸는 함수임.
    ex) python -> 파이썬

    """

    result: list = words.split()

    eng_list = englist["eng"].tolist()

    for i in range(len(result)):
        lower_case = result[i].lower()
        if lower_case in eng_list:

            eng_to_kor = englist[englist["eng"] == lower_case]["kor"].values[0]
            if print_on:
                print("변환 :", lower_case, " => ", eng_to_kor)

            result[i] = eng_to_kor
    return result


def find_han(text: str) -> list:
    """
    문장 내 한글만 추출
    """
    return re.findall("[\u3130-\u318F\uAC00-\uD7A3]+", text)


def find_eng(text: str, min_num=3) -> list:
    """
    문장 내 영어만 추출
    """
    x = re.findall("[a-zA-Z]+", text)
    x = [i.lower() for i in x if len(i) > min_num]
    return x


def key_extraction(
    token_list: list, model, min_num=3, min_rank=20, Noun_extractor=Hannanum()
) -> pd.DataFrame:
    """
    문서의 핵심 키워드를 추출하는 매서드
    1. 문장 내 단어 출현 횟수 계산 및 리스트 확보
    2. sentence_transformers 활용 token_list, candidate_wrds 임베딩
    3. 문장(1개)과 단어(N개)의 cosine similarity 측정
    4. 연관도가 높은 단어 순으로 검색결과 제공

    *-- param --*

    token_list : 토큰화 된 리스트 ex) ['파이썬','라이브러리','예제입니다.']
    model : sentence_transformers 모델
    min_num = 키워드 최소 출현 횟수
    min_rank = 연관성 있는 키워드 추출 개수

    """
    # 문장 내 단어에 대한 value_counts

    # str or list에 따른 tokenizing 방법
    if type(token_list) == list:
        han_nouns = Noun_extractor.nouns(" ".join(token_list))

    else:
        print("book_info type must be list.")

    candidates = pd.DataFrame(han_nouns)[0].value_counts()

    # min_num개 이상인 단어만 추출
    candidate_words = candidates[candidates >= min_num].index.values.tolist()

    doc_embedding = model.encode([" ".join(token_list)])
    candidate_embeddings = model.encode(candidate_words)
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    return pd.DataFrame(distances.T, index=candidate_words).sort_values(by=0, ascending=False)[
        :min_rank
    ]
