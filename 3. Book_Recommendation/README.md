# Sentence Bert를 활용해 연관성 높은 도서 추천하기

- 프로젝트의 목적은 Cross encoder와 Bi Encoder를 활용해 문서 추천 기능을 구현하고 문서 embedding을 활용하는 방법을 학습하는 데 있음.

- 특정 도서와 연관성 높은 도서를 추천하는 시스템을 Sentence bert를 활용해 구현하였음.

- 추천 시스템은 1. 선택한 도서(Query)와 전체 도서(Document Collection)를 Bi Encoder로 비교하여 1차로 후보군을 추출한 다음 2. Cross Encoder를 활용해 추천 순위를 결정하는 방식으로 진행됨.

    <img src='../img/InformationRetrieval.png'>

## 문서 추천 예시

> 도서 제목 : 그림과 실습으로 배우는 도커 & 쿠버네티스

<br/>

- Bi Encoder로 1차 후보군 선정

    <table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> <th></th> <th>book_title</th> </tr> </thead> <tbody> <tr> <th>1009</th> <td>따라하며 배우는 도커와 CI 환경</td> </tr> <tr> <th>3287</th> <td>리눅스 서버 관리 바이블</td> </tr> <tr> <th>1368</th> <td>타입스크립트, AWS 서버리스로 들어올리다</td> </tr> <tr> <th>419</th> <td>초보를 위한 젠킨스 2 활용 가이드 2/e</td> </tr> <tr> <th>3274</th> <td>오픈스택을 다루는 기술</td> </tr> <tr> <th>4713</th> <td>Windows Server Container 시작하기</td> </tr> <tr> <th>3580</th> <td>도커, 컨테이너 빌드업!</td> </tr> <tr> <th>4337</th> <td>Windows Server 2016 Hyper-V 쿡북 2/e</td> </tr> <tr> <th>3535</th> <td>빠르게 훑어보는 구글 클라우드 플랫폼</td> </tr> <tr> <th>3358</th> <td>파이썬을 이용한 머신러닝, 딥러닝 실전 개발 입문</td> </tr> </tbody></table>

<br/>

- Cross Encoder로 도서 추천 순위 계산

  ```python

      [
      (0.76013, '초보를 위한 젠킨스 2 활용 가이드 2/e'),
      (0.75422, '도커, 컨테이너 빌드업!'),
      (0.73951, '따라하며 배우는 도커와 CI 환경'),
      (0.73613, '헬름 배우기'),
      (0.72709, '서비스 운영이 쉬워지는 AWS 인프라 구축 가이드')
      ]

  ```
