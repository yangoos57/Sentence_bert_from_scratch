from gensim.models import Word2Vec
from typing import Iterable, List
from kiwipiepy import Kiwi
from itertools import chain
import pandas as pd
import logging


class W2VTrainer:
    """
    W2V 모델 학습 및 학습에 필요한 데이터 정제를 수행하는 클래스 입니다.

    Parameter
    ---------

    - dir: 영어 단어 -> 한국어 단어 또는 오탈자 -> 정상 단어로 변환하기 위해 사용하는 파일을 불러옵니다.
         ex) python -> 파이썬 || 파이선 -> 파이썬

    """

    def __init__(self, dir: str = None) -> None:
        """형태소 분석기인 kiwi 실행"""
        # noun extractor
        self.noun_extractor = Kiwi(model_type="knlm")
        self.dir = dir if dir else "../../data/preprocess/eng_han.csv"
        self._update_noun_words()

        self.logging = logging("logs/check_process.log", __name__)

    def _update_noun_words(self):
        """Kiwi에 등록되지 않은 단어 추가"""
        eng_han_df = pd.read_csv(self.dir)
        han_words = eng_han_df
        for val in han_words.kor.values:
            self.noun_extractor.add_user_word(val)

    def train_and_save(self, df: pd.DataFrame, dir: str, min_length: int = 2) -> Word2Vec:
        """
        W2V 학습용 데이터를 생성하고 이를 활용해 모델을 학습하는 메서드입니다.

        Parameter
        --------
        df: 학습 데이터는 pd.Dataframe 형식만을 지원합니다.
        min_length: 학습 데어티에 포함할 단어의 최소 길이를 정의합니다.

        """
        self.logging.info("start create_w2v_data")
        w2v_data = self.create_w2v_data(df, min_length)
        self.logging.info("start training")
        embedding_model = Word2Vec(sentences=w2v_data, window=2, min_count=30, workers=7, sg=1)
        self.logging.info("save model")
        embedding_model.wv.save_word2vec_format(dir)
        return None

    def create_w2v_data(self, df: pd.DataFrame, min_length: int = 2) -> List[list]:
        """
        W2V 모델 학습에 활용될 데이터를 생성하는 메서드입니다.

        Parameter
        --------
        df: 학습 데이터는 pd.Dataframe 형식만을 지원합니다.
        min_length: 학습 데어티에 포함할 단어의 최소 길이를 정의합니다.

        """
        list_data = map(lambda x: self._convert_series_to_str(x[1]), df.iterrows())
        noun_data = self._extract_noun_data(list_data)
        refined_data = map(self._map_english_to_hangeul, noun_data)
        w2v_data = map(lambda x: self._eliminate_min_len_word(x, min_length), refined_data)
        return list(w2v_data)

    def _convert_series_to_str(self, series: pd.Series) -> str:
        """Series에 속한 값을 하나의 str으로 연결"""
        book_title = series["title"]
        series = series.drop(["title", "isbn13"])
        return book_title + " " + " ".join(list(chain(*series.values)))

    def _extract_noun_data(self, word_iter: Iterable[str]) -> List[List[str]]:
        """연결된 str을 형태소 분석하여 한글 명사 및 영단어 추출"""
        tokenized_words = self.noun_extractor.tokenize(word_iter)
        result = []
        for lst in tokenized_words:
            words = [word.form for word in lst if word.tag in ("NNG", "NNP", "SL")]
            result.append(words)
        return result

    def _map_english_to_hangeul(self, word_list: List[str]) -> List[str]:
        """영단어를 한국어 단어로 치환"""
        eng_han_df = pd.read_csv(self.dir).dropna()
        eng_han_dict = dict(eng_han_df.values)

        def _map_eng_to_han(word: str, eng_han_dict: dict) -> str:
            han_word = eng_han_dict.get(word)
            return han_word if han_word else word

        return list(map(lambda x: _map_eng_to_han(x.lower(), eng_han_dict), word_list))

    def _eliminate_min_len_word(self, data, min_length: int = 2) -> List[str]:
        """단어 길이가 min_length 이하인 경우 제거"""
        return list(filter(lambda x: len(x) >= min_length, data))
