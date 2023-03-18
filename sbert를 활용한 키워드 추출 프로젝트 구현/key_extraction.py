from model import SentenceBert
from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraModel, ElectraTokenizerFast
from typing import Union, Tuple, List, Dict
from collections import Counter
from kiwipiepy import Kiwi
from itertools import chain, islice
import pandas as pd
import numpy as np
import torch


class keywordExtractor:
    """
    Encoder 기반 모델을 활용해 모델의 키워드를 추출하는 클래스입니다.

    Parameter
    --------
    - model: Encoder 기반 언어모델을 사용합니다. 기본 값으로 "monologg/koelectra-base-v3-discriminator"를 사용하고 있습니다.
    tokenizer: 해당 모델에 맞는 토크나이저를 사용합니다.

    - dir: 영어 단어 -> 한국어 단어 또는 오탈자 -> 정상 단어로 변환하기 위해 사용하는 파일을 불러옵니다.
         ex) python -> 파이썬 || 파이선 -> 파이썬

    """

    def __init__(self, model=None, tokenizer=None, dir: str = None) -> None:
        """언어모델 및 형태소분석기 불러오기"""

        # models
        name = "monologg/koelectra-base-v3-discriminator"
        self.model = model if model else ElectraModel.from_pretrained(name)
        self.tokenizer = tokenizer if tokenizer else ElectraTokenizerFast.from_pretrained(name)
        self.sbert = SentenceBert(self.model)
        self.sbert.eval()

        # noun extractor
        self.noun_extractor = Kiwi(model_type="knlm")
        self.dir = dir if dir else "../../data/preprocess/eng_han.csv"

        # mapper
        self.eng_kor_df = pd.read_csv(dir)
        self._update_noun_words()

    def _update_noun_words(self):
        """Kiwi에 등록되지 않은 단어 추가"""
        kor_words = self.eng_kor_df
        for val in kor_words.kor.values:
            self.noun_extractor.add_user_word(val)

    def extract_keyword_list(self, doc: pd.Series, min_count: int = 3, min_length: int = 2) -> List:
        """
        min_count 이상 집계, 단어 길이 최소 min_length 이상인 단어를 수집합니다.

        parameter
        ---------------
        - doc: 도서정보
        - min_count: 문장 내 최소 출현 빈도
        - min_length: 단어의 최소 길이 min_length=2 설정 시 한 글자인 단어 제거

        """
        raw_data = self._convert_series_to_list(doc)
        keyword_list = self._extract_keywords(raw_data)
        translated_keyword_list = self._map_english_to_korean(keyword_list)
        refined_keyword_list = self._eliminate_min_count_words(translated_keyword_list, min_count)
        return list(filter(lambda x: len(x) >= min_length, refined_keyword_list))

    def _convert_series_to_list(self, series: pd.Series) -> List[List[str]]:
        """Series에 속한 값을 하나의 str으로 연결"""
        book_title = series["title"]
        series = series.drop(["title", "isbn13"])

        raw_data = [book_title] + list(chain(*series.values))
        return list(chain(*map(lambda x: x.split(), raw_data)))

    def _extract_keywords(self, words: List[str]) -> List[List[str]]:
        """연결된 str을 형태소 분석하여 한글 명사 및 영단어 추출"""
        tokenized_words = self.noun_extractor.tokenize(" ".join(words))
        return [word.form for word in tokenized_words if word.tag in ("NNG", "NNP", "SL")]

    def _map_english_to_korean(self, word_list: list[str]) -> list[str]:
        """영단어를 한국어 단어로 치환"""

        converter = dict(self.eng_kor_df.dropna().values)

        def map_eng_to_kor(word: str) -> str:
            kor_word = converter.get(word)
            return kor_word if kor_word else word

        return list(map(lambda x: map_eng_to_kor(x.lower()), word_list))

    def _eliminate_min_count_words(self, candidate_keyword, min_count: int = 3):
        """min_count 이상으로 집계되지 않은 단어 제거"""
        refined_kor_words = filter(lambda x: x[1] >= min_count, Counter(candidate_keyword).items())
        return list(map(lambda x: x[0], refined_kor_words))

    def create_keyword_embedding(self, doc: pd.Series) -> torch.Tensor:
        """
        keyword embedding를 생성하는 메서드입니다.

        parameter
        --------
        - doc : pd.Series 데이터
        """
        keyword_list = self.extract_keyword_list(doc)
        tokenized_keyword = self.tokenize_keyword(keyword_list)
        return self._create_keyword_embedding(tokenized_keyword)

    def tokenize_keyword(self, text: Union[list[str], str], max_length=128) -> Dict:
        if text:
            pass
        else:
            text = ["에러"]
        token = self.tokenizer(
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

    def _create_keyword_embedding(self, tokenized_keyword: dict) -> torch.Tensor:

        # extract attention_mask, keyword_embedding
        attention_mask = tokenized_keyword["attention_mask"]
        keyword_embedding = self.model(**tokenized_keyword)["last_hidden_state"]

        # optimize attention_mask, keyword_embedding
        attention_mask, optimized_keyword_embedding = self._delete_cls_sep(
            attention_mask, keyword_embedding
        )

        # mean pooling
        keyword_embedding = self._pool_keyword_embedding(
            attention_mask, optimized_keyword_embedding
        )

        return keyword_embedding

    def _delete_cls_sep(
        self, attention_mask: torch.Tensor, keyword_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[CLS],[SEP] 토큰 제거"""
        attention_mask = attention_mask.detach().clone()
        keyword_embedding = keyword_embedding.detach().clone()

        # delete [cls], [sep] in attention_mask
        num_keyword = attention_mask.size(0)
        for i in range(num_keyword):
            sep_idx = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
            attention_mask[i][0] = 0  # [CLS] => 0
            attention_mask[i][sep_idx] = 0  # [SEP] => 0

        # delete [cls], [sep] in keyword_embedding
        boolean_mask = attention_mask.unsqueeze(-1).expand(keyword_embedding.size()).float()
        keyword_embedding = keyword_embedding * boolean_mask
        return attention_mask, keyword_embedding

    def _pool_keyword_embedding(
        self, attention_mask: torch.Tensor, keyword_embedding: torch.Tensor
    ) -> torch.Tensor:
        """keyword embedding에 대해 mean_pooling 수행"""

        num_of_tokens = attention_mask.unsqueeze(-1).expand(keyword_embedding.size()).float()
        total_num_of_tokens = num_of_tokens.sum(1)
        total_num_of_tokens = torch.clamp(total_num_of_tokens, min=1e-9)

        sum_embeddings = torch.sum(keyword_embedding, 1)

        # Mean Pooling
        mean_pooling = sum_embeddings / total_num_of_tokens
        return mean_pooling

    def create_doc_embedding(self, doc: pd.Series) -> torch.Tensor:
        """sbert를 활용해 doc_embedding 생성"""
        stringified_doc = self._convert_series_to_str(doc)
        tokenized_doc = self.tokenize_keyword(stringified_doc)
        return self._create_doc_embedding(tokenized_doc)

    def _convert_series_to_str(self, series: pd.Series) -> str:
        """Series에 속한 값을 하나의 str으로 연결"""
        book_title = series["title"]
        series = series.drop(["title", "isbn13"])
        return book_title + " " + " ".join(list(chain(*series.values)))

    def _create_doc_embedding(self, tokenized_doc: Union[list[str], str]) -> torch.Tensor:
        """sbert를 활용해 doc_embedding 생성"""
        return self.sbert(**tokenized_doc)["sentence_embedding"]

    def extract_keyword(self, docs: pd.DataFrame) -> Dict:
        """
        도서 데이터를 기반으로 키워드를 추출하는 메서드입니다.
        클래스 내 create_keyword_list, create_keyword_embedding, create_doc_emeddings를 기반으로 동작합니다.

        Parameter
        ---------
        - docs : pd.DataFrame 타입의 데이터이며 column은 [isbn13, title, toc, intro, publisher]이어야 합니다.

        """
        if docs.columns.tolist() != ["isbn13", "title", "toc", "intro", "publisher"]:
            raise ValueError(
                f"{docs.columns.tolist()} doesn't match with ['isbn13', 'title', 'toc', 'intro', 'publisher']"
            )

        keyword_embedding = map(lambda x: self.create_keyword_embedding(x[1]), docs.iterrows())
        doc_embedding = map(lambda x: self.create_doc_embedding(x[1]), docs.iterrows())
        keyword_list = map(lambda x: self.extract_keyword_list(x[1]), docs.iterrows())

        co_sim_score = map(
            lambda x: self._calc_cosine_similarity(*x).flatten(),
            zip(doc_embedding, keyword_embedding),
        )
        top_n_keyword = list(
            map(lambda x: self._filter_top_n_keyword(*x), zip(keyword_list, co_sim_score))
        )

        return dict(isbn13=docs["isbn13"].tolist(), keywords=top_n_keyword)

    def _calc_cosine_similarity(
        self, doc_embedding: torch.Tensor, keyword_embedding: torch.Tensor
    ) -> np.array:
        """단어와 문장 간 코사인 유사도 계산"""

        doc_embedding = doc_embedding.detach()
        keyword_embedding = keyword_embedding.detach()

        doc_score = list(
            map(lambda x: cosine_similarity(x.unsqueeze(0), keyword_embedding), doc_embedding)
        )

        max_pooling = np.max(doc_score, axis=0)  # Max
        return max_pooling

    def _filter_top_n_keyword(
        self, keyword_list: List, co_sim_score: np.array, rank: int = 20
    ) -> List:
        """top_n 키워드 추출"""
        keyword = dict(zip(keyword_list, co_sim_score))
        sorted_keyword = sorted(keyword.items(), key=lambda k: k[1], reverse=True)
        return list(dict(islice(sorted_keyword, rank)).keys())
