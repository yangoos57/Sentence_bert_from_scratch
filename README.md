# Sentence Bert from scratch

- `Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks`논문을 코드로 구현하고, Sentence Bert를 활용해 문서 키워드 추출과 연관성 도서 추천 기능에 대한 2건의 미니프로젝트를 수행하였음.

<br/>

## 프로젝트 상세

**1.Bi_encoder & Cross_Encoder**

- Bi-encoder와 Cross-Encoder를 코드로 구현하고, 논문을 코드 기반으로 이해할 수 있도록 튜토리얼을 작성하였음.

  <br/>

  <img src='img/Bi_vs_Cross-Encoder.png' alt='Bi_vs_Cross-Encoder' width='400px'>

  <em>사진 출처 : sbert.net</em>

  <br/>

**2.keyword_extraction_for_docs**

- Bi Encoder를 활용해 문서의 핵심 키워드를 추출하는 시스템을 구현하였음.

  <img src='img/key_extraction.png' alt='key_extraction' width='400px'>

  <br/>

**3.Book_Recommendation**

- Cross Encoder와 Bi Encoder를 활용해 특정 도서(Query)와 연관있는 도서를 추천하는 시스템을 구현하였음.

  <img src='img/InformationRetrieval.png' width = '600px'>

  <em>사진 출처 : sbert.net</em>

<br />

## 참고한 라이브러리

- [sentence Bert](https://github.com/UKPLab/sentence-transformers)

- [KeyBert](https://github.com/MaartenGr/KeyBERT)
