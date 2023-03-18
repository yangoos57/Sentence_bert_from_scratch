### 데이터 정보

- w2v : word2vec 용 데이터

- data_for_search : List

  - 자료 검색용 데이터
  - isbn, keyword

- book_info.parquet : pd.DataFrame

  - 정보나루 API에서 스크랩한 장서 데이터
  - isbn13, bookname, authors, publisher, class_no, reg_date, bookImageURL

- scraping_result.parquet : pd.DataFrame

  - 교보문고에서 스크랩한 장서 데이터
  - isbn13, title, toc, intro, publisher
