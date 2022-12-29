import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class modelWithPooling(nn.Module):
    def __init__(self, model, pooling_type="mean") -> None:
        super().__init__()

        self.model = model  # base model ex)BertModel, ElectraModel ...
        self.pooling_type = pooling_type  # pooling type 선정
        self.tokenizer = None

    def encode(self, items: list, tokenizer=None, batch_size: int = 16):

        if tokenizer is not None:
            self.tokenizer = tokenizer

        if self.tokenizer is None:
            from transformers import AutoTokenizer

            print(f'Loading Tokenizer : "{self.model.config._name_or_path}"')
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model.config._name_or_path
            )

        def default_collater(items):
            token = self.tokenizer(
                items, padding=True, truncation=True, return_tensors="pt"
            )
            return {"sen": items, "token": token}

        data_loader = DataLoader(
            dataset=items, batch_size=batch_size, collate_fn=default_collater
        )

        output_lst = []
        sen_lst = []
        for data in data_loader:
            sen = data.pop("sen")
            sen_lst += sen
            token = data.pop("token")
            outputs = self.forward(**token)["sentence_embedding"]
            output_lst.append(outputs)

        return {"sen": sen_lst, "sentence_embedding": torch.cat(output_lst)}

    def forward(self, **kwargs):
        features = self.model(**kwargs)
        # [batch_size, src_token, embed_size]
        attention_mask = kwargs["attention_mask"]

        last_hidden_state = features["last_hidden_state"]

        if self.pooling_type == "cls":
            """
            [cls] 부분만 추출
            """

            cls_token = last_hidden_state[:, 0]  # [batch_size, embed_size]
            result = cls_token

        if self.pooling_type == "max":
            """
            문장 내 토큰 중 가장 값이 큰 token만 추출
            """

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # Set padding tokens to large negative value
            last_hidden_state[input_mask_expanded == 0] = -1e9
            max_over_time = torch.max(last_hidden_state, 1)[0]
            result = max_over_time

        if self.pooling_type == "mean":
            """
            문장 내 토큰을 합한 뒤 평균
            """
            # padding 부분 찾기 = [batch_size, src_token, embed_size]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            # padding인 경우 0 아닌 경우 1곱한 뒤 총합 = [batch_size, embed_size]
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)

            # 평균 내기위한 token 개수
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            result = sum_embeddings / sum_mask

        #  input.shape : [batch_size, src_token, embed_size] => output.shape : [batch_size, embed_size]
        return {"sentence_embedding": result}
