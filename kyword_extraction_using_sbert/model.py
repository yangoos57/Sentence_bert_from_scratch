from transformers import ElectraModel, ElectraTokenizerFast
from torch.utils.data import DataLoader
from typing import List, Dict
import torch.nn as nn
import torch


class SentenceBert(nn.Module):
    """
    Sentence Bert 논문을 읽고 관련 Repo를 참고해 만든 모델입니다.

    Huggingface Trainer API를 쉽게 활용할 수 있도록 모델을 일부 수정했습니다.

    Parameter
    ---------
    - model : Huggingface에서 제공하는 BaseModel을 활용해야 합니다.
    - tokenizer : 모델에 맞는 토크나이저를 사용해야하며 TokenizerFast를 통해 불러와야합니다.
    - model 및 tokenizer 설정이 없는 경우 "monologg/koelectra-base-v3-discriminator" 를 기본 모델로 사용합니다.
    - pooling_type : 논문에서 제시하는 Pooling 방법인 mean pooling, max pooling, CLS pooling을 지원하며 기본 설정 값은 mean입니다.

    """

    def __init__(self, model=None, pooling_type: str = "mean") -> None:
        super().__init__()
        name = "monologg/koelectra-base-v3-discriminator"
        self.model = model if model else ElectraModel.from_pretrained(name)

        if pooling_type in ["mean", "max", "cls"] and type(pooling_type) == str:
            self.pooling_type = pooling_type
        else:
            raise ValueError("'pooling_type' only ['mean','max','cls'] possible")

    def forward(self, **kwargs):
        attention_mask = kwargs["attention_mask"]
        last_hidden_state = self.model(**kwargs)["last_hidden_state"]

        if self.pooling_type == "cls":
            """[cls] token을 sentence embedding으로 활용"""
            result = last_hidden_state[:, 0]

        if self.pooling_type == "max":
            """문장 내 여러 토큰 중 가장 값이 큰 token만 추출하여 sentence embedding으로 활용"""

            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[num_of_tokens == 0] = -1e9
            result = torch.max(last_hidden_state, 1)[0]

        if self.pooling_type == "mean":
            """문장 내 토큰을 평균하여 sentence embedding으로 활용"""

            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

            sum_embeddings = torch.sum(last_hidden_state * num_of_tokens, 1)

            total_num_of_tokens = num_of_tokens.sum(1)
            total_num_of_tokens = torch.clamp(total_num_of_tokens, min=1e-9)

            result = sum_embeddings / total_num_of_tokens

        return {"sentence_embedding": result}
