from transformers import ElectraTokenizerFast, ElectraModel, ElectraForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Dict
from model import SentenceBert
from itertools import chain
import pandas as pd
import numpy as np
import torch
from ast import literal_eval


class BookRecommander:
    def __init__(self, bi_encoder, cross_encoder, tokenizer, dir) -> None:
        self.bi_encoder = (
            bi_encoder if bi_encoder else ElectraModel.from_pretrained("./model/bi_encoder").eval()
        )
        self.cross_encoder = (
            cross_encoder
            if cross_encoder
            else ElectraForSequenceClassification.from_pretrained("./model/cross_encoder").eval()
        )
        self.tokenizer = (
            tokenizer
            if tokenizer
            else ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
        )

        self.dir = dir if dir else "data/preprocess/eng_han.csv"
        self.eng_kor_df = pd.read_csv(dir)

    def _convert_series_to_str(self, series: pd.Series) -> str:
        """Series에 속한 값을 하나의 str으로 연결"""
        book_title = series["title"]
        series = series.drop(["title", "isbn13"])
        return book_title + " " + " ".join(list(chain(*series.values)))

    def tokenize_words(text: Union[List[str], str], tokenizer, max_length=128) -> Dict:
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

    def _map_english_to_korean(self, word_list: list[str]) -> list[str]:
        """영단어를 한국어 단어로 치환"""

        converter = dict(self.eng_kor_df.dropna().values)

        def map_eng_to_kor(word: str) -> str:
            kor_word = converter.get(word)
            return kor_word if kor_word else word

        return list(map(lambda x: map_eng_to_kor(x.lower()), word_list))

    def _load_book_embedding(self):
        lst = []
        for i in range(1, 6):
            x = torch.load(
                f"book_embeddings/bi_encoder_doc_{i}.pt", map_location=torch.device("cpu")
            )
            lst.append(pd.DataFrame(x))
        return pd.concat(lst, axis=0).reset_index(drop=True)

    def extract_candidates(self, query_info: pd.Series) -> pd.DataFrame:
        query_info_str = self._convert_series_to_str(query_info)
        translated_query_info_str = self._map_english_to_korean(query_info_str)
        tokenized_query_info = self.tokenize_words(translated_query_info_str)

        query_sentence_embedding = self.bi_encoder(**tokenized_query_info)[
            "sentence_embedding"
        ].detach()
        query_doc_embedding = torch.sum(
            query_sentence_embedding, dim=0
        ) / query_sentence_embedding.size(0)

        total_book_doc_list = self.load_book_embedding()
        total_book_doc_embedding = torch.cat(total_book_doc_list["rep_embedding"].values.tolist())

        cos_sim = cosine_similarity(query_doc_embedding.unsqueeze(0), total_book_doc_embedding)
        top_ten_idx = np.argsort(cos_sim)[0][::-1][:20]

        pd.DataFrame(raw_data.iloc[top_ten_idx]["book_title"]).reset_index(drop=True)[1:]

    def rerank_candidates(
        self,
        query: pd.Series,
        candidates: pd.DataFrame,
        top_k=10,
        max_length=256,
        sentence_n=4,
    ) -> List:
        """
        query : 찾고자 하는 도서 임베딩
        candidates : 1차 후보 도서 정보
        eng_kor_df : 한,영 단어집
        tokenizer: 토크나이저
        top_k : 상위 k개 도서 정보만 제한해서 추출
        max_length : 도서 비교 시 활용될 문장의 길이
        sentence_n : 도서 비교 시 활용될 문장 개수
        """

        lst = []
        for i in range(1, len(candidates)):

            # 후보군 전처리
            candidate = self._convert_series_to_str(candidates.iloc[i])
            candidate = self.map_english_to_korean(candidate, self.eng_kor_df)
            candidate = " ".join(candidate)

            # tokenize query and candidate
            token = self.tokenizer(
                query,
                candidate,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
                stride=20,
                return_overflowing_tokens=True,
            )
            token.pop("overflow_to_sample_mapping")

            # 도서 비교에 활용 될 문장 샘플링
            torch.manual_seed(42)

            tokens = token.input_ids
            random_int = torch.randint(0, tokens.size(0), size=(sentence_n,))
            tokens = tokens[random_int]

            # Cross Encoder로 문장 유사도 추출
            logit = self.cross_encoder(tokens)["logits"].detach()
            lst.append((round(torch.mean(logit).item(), 5), candidates.iloc[i]["title"]))

            # 연관성 높은 순으로 재졍렬
        lst = sorted(lst, key=lambda x: x[0])[::-1]

        return lst[:top_k]
