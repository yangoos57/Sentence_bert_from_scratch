from transformers import ElectraModel, ElectraTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from model import SentenceBert
from random import randrange
from konlpy.tag import Hannanum
import pandas as pd
import numpy as np
import torch
import utils


def tokenizing_function(text, max_length=128):
    token = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        stride=20,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )
    token.pop("overflow_to_sample_mapping")
    return token


def keyword_extraction_from_doc(
    doc: str, sbert, min_num=3, num_rank=20, max_length=128, noun_extractor=Hannanum()
):
    """
    문서 핵심 키워드를 추출하는 매서드
    1단계. max_length개 토큰 이상으로 구성된 문서인 경우 여러 문서로 나누어짐
    2단계. 문서 내 한글 및 영문 키워드 추출
    3단계. Sbert 활용 문서 및 키워드 임베딩
    4단계. 문서와 키워드 간 cosine similarity 측정
    5단계. 연관도가 높은 단어 순으로 검색결과 제공

    *-- params --*

    doc : 문서 정보
    sbert : sbert모델 불러오기
    min_num : 키워드 최소 출현 횟수 설정 ex) 문서 내 3회 이상 사용된 경우 추출
    min_rank : 키워드 순위 설정
    max_length : 문장 Tokenizing 범위
    noun_extractor : 한글 명사 추출을 위한 전처리 라이브러리(Konlpy)활용
    """

    # Doc embedding
    # Doc Token이 128개 이상인 경우 여러 문장으로 구분
    # ex) 1280개 토큰이 있는 Doc인 경우 128개 토큰이 있는 문장 10개로 구분

    token = tokenizing_function(doc, max_length=max_length)
    logits_sentence = sbert(**token)["sentence_embedding"]

    # Keyword 사용될 단어 추출
    #
    # 한국어 명사 추출
    if type(doc) == str:
        han_nouns = noun_extractor.nouns(doc)
    else:
        raise TypeError("doc must be str type.")

    # 한글 키워드 추출
    candidates_kor = pd.DataFrame(han_nouns)[0].value_counts()
    candidate_kor_words = candidates_kor[candidates_kor >= min_num].index.values.tolist()
    candidate_kor_words = [i for i in candidate_kor_words if len(i) > 1]

    # Doc 내 영문 추출
    book_info_eng = utils.find_eng(doc, min_num=0)

    # 영문 키워드 추출
    if book_info_eng:
        candidates_eng = pd.DataFrame(book_info_eng)[0].value_counts()
        candidates_eng_words = candidates_eng[candidates_eng >= min_num].index.values.tolist()
    else:
        candidates_eng_words = []

    # 키워드 총합
    candidate_words = candidate_kor_words + candidates_eng_words

    if candidate_words == False:
        return pd.DataFrame(columns=["유사도"])

    # Keyword Embedding
    token_embedding = tokenizing_function(candidate_words)
    last_hidden_state = model(**token_embedding)["last_hidden_state"]

    # [CLS], [SEP] 제거 ([CLS], [SEP]을 제거하면 정확도가 올라감.)
    attention_mask = token_embedding["attention_mask"]
    for i in range(attention_mask.size(0)):
        # x = attention mask 1에 포함 된 마지막 index
        x = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
        attention_mask[i][0] = 0  # [CLS] = 0
        attention_mask[i][x] = 0  # [SEP] = 0

    # 토큰 내 padding 부분 찾기 = [batch_size, src_token, embed_size]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # padding인 경우 0 아닌 경우 1을 곱함 = [batch_size, embed_size]
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

    # 평균을 위한 token 개수 확대
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)

    # Mean Pooling
    result = sum_embeddings / sum_mask

    # Keyword와 Doc 문장 비교
    sentence_comparsion = []
    len_sentences = logits_sentence.size(0)
    for i in range(len_sentences):
        sentence_comparsion.append(
            cosine_similarity(logits_sentence[i].unsqueeze(0).detach(), result.detach())
        )

    # Mean Pooling
    # result =  np.sum(sentence_comparsion,axis=0)/len_sentences # Mean

    # Max Pooling
    result = np.max(sentence_comparsion, axis=0)  # Max

    # Ranking 시각화 Return
    return pd.DataFrame(result.T, index=candidate_words, columns=["유사도"]).sort_values(
        by="유사도", ascending=False
    )[:num_rank]


if __name__ == "__main__":

    model = ElectraModel.from_pretrained("../model/disc_book_final")
    tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")

    model_with_pooling = SentenceBert(model, tokenizer)
    raw_data = pd.read_csv("../data/bookList/raw_book_info_list.csv", index_col=0)
    englist = pd.read_csv("../data/preprocess/englist.csv")

    # Row에 있는 모든 item을 하나의 Str으로 변환
    num = randrange(len(raw_data))
    book_info: str = utils.merge_series_to_str(raw_data.iloc[num])

    # 영단어 일부를 한글로 변환 ex) Python => 파이썬
    book_info_trans = utils.trans_eng_to_han(book_info, englist=englist)

    print(keyword_extraction_from_doc(" ".join(book_info_trans), model_with_pooling, num_rank=20))
