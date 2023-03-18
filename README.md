# Sentence Bert from scratch

- `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks`논문을 코드로 구현하였음.

- Sentence Bert를 활용해 문서 키워드 추출 모델을 구현하고 이를 활용해 도서 추천 모델을 개발하였음.

<br/>

## 프로젝트 상세

**1.Bi_Encoder & Cross_Encoder**

- Bi-Encoder와 Cross-Encoder를 코드로 구현하고, 논문을 코드 기반으로 이해할 수 있도록 튜토리얼을 작성하였음.

  <br/>

  <img src='images/Bi_vs_Cross-Encoder.png' alt='Bi_vs_Cross-Encoder' width='400px'>

  <em>사진 출처 : sbert.net</em>

  <br/>

**2.keyword_extraction_for_docs**

- Bi-Encoder를 활용해 문서의 핵심 키워드를 추출하는 시스템을 구현하였음.

  <img src='images/key_extraction.png' alt='key_extraction' width='400px'>

- 이렇게 제작한 모델을 활용해 도도모아 프로젝트를 수행하였음.

## 참고한 라이브러리

- [sentence Bert](https://github.com/UKPLab/sentence-transformers)

- [KeyBert](https://github.com/MaartenGr/KeyBERT)
