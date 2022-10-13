## 2. MDS

**MDS는 Multidimensional Scaling의 약자로, Unsupervised learning 기반의 선형 변수 추출 방법**이다. 이는 원 공간에서 모든 점들 간에 정의된 거리 행렬 d가 주어졌을 때, 임베딩 공간에서의 **Euclidean distance 인 $|y_{i} - y_{j} |$ 와 거리 행렬의 차이가 최소가 되는 임베딩 y**를 학습한다.

즉, **객체간 거리 정보**를 유지하는 것을 목적으로 하는데, 고차원 상에서 개체간 거리를 동일하게 유지하면서 저 차원 공간으로 매핑하는 것.
