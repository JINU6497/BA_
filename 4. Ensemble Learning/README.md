# 목차

1. Why Ensemble ?
2. Bagging, Boosting
3. Extreme Gradient Boosting Machine (XGBoost)
4. LightGBM
5. CatBoost
6. Tutorial

## 1. Why Ensemble ?

이번 시간에는 **Ensemble** 에 대해서 알아보도록 하겠습니다. 

**“There’s No Free Lunch”**  라는 말이 있습니다. 분야마다 다르게 사용되는 말일 수 있지만, 저희 같은 Data science들이 사용할 경우에는, 하나의 알고리즘이 모든 데이터에 대해서 모두 잘 동작하지는 않는다는 것입니다. 그 이유는 다음과 같습니다.

- Model은 어떠한 가정을 기반으로 하여 수립된다.
- 그렇지만, 그런 가정은 특정 경우에는 잘 동작하지 않는다.

이러한 이유가 있기에, 저희는 최대한 많은 알고리즘을 공부하여, 상황에 맞게 잘 사용할 수 있어야 합니다.

그렇다면 왜 가정은 특정 경우에는 잘 동작하지 않는 것일까요?

![image](https://user-images.githubusercontent.com/87464956/204961598-299a7ae1-4606-421b-8d9a-d912230ba11d.png)


다음과 같은 2가지 경우를 생각해봅시다.

먼저, 모두 알다시피 현실 세계에 존재하는 데이터에는 항상 **Noise $ε$가 존재**합니다. 이에 항상 정확한 추정이 불가능하고, 정확한 추정을 하게 되어도 다른 현실 세계의 데이터에는 또 다른 Noise $ε$가 존재하기에 오히려 더 좋지 못한 성능을 보일 수 있습니다. 

다음으로 모델을 생각해봅시다. 현실 세계에 **Noise $ε$** 가 껴있는 만큼 **모델의 성능, 즉 Generalization Performance는 더 떨어질 것**입니다. 이때, 모델이 Generalization 하지 않게 되는 Error는 다음과 같이 크게 **Bias Error**와, **Variance Error**로 나뉠 수 있습니다.

먼저 **Bias Error**는 반복적으로 모델을 학습할 경우, 평균적으로 얼마나 정확한 추정이 가능한지를 추정하는 지표이고, **Variance Error**는 반복적으로 모델을 학습할 경우, 개별 추정이 얼마나 정확한지를 측정하는 지표입니다.

이때, 저희는 다수의 모델을 이용하여 Data의 Noise에 따른 Model의 Bias와 Variance를 줄이고자 합니다. 이때 사용하는 것이 바로 **Ensemble**입니다.

## 2. Bagging, Boosting

![image](https://user-images.githubusercontent.com/87464956/204961625-288f4d65-6b1d-4f64-8257-4f8a58a252cc.png)

위에서 먼저 말했듯이, **Ensemble은 여러 개의 Weak 모델들을 합쳐서 하나의 Strong 모델을 만드는 것을 목적**으로 합니다. 이때 크게 다음과 같은 2가지의 방식이 존재합니다.

- **Bagging:** 개별 모델에 데이터를 Split하여 집어넣은 후, 결과를 Aggregate 하는 방식으로 진행. 이러한 방식을 분산이 높고, 편향이 낮은 알고리즘에 적용한다면 **Complexity가 높은 모델의 Variance Error를 감소** 시킬 수 있음
- **Boosting:** 하나의 Weak model에서 시작하여 다음 모델을 만드는데, 이전 시점의 모델보다 더 나은 모델을 만드는 방식. 이러한 방식을 분산이 낮고 편향이 높은 알고리즘에 적용한다면 **Complexity가 낮은 모델의 Bias Error를 감소** 시킬 수 있음.

## 3. Extreme Gradient Boosting Machine (XGBoost)

![image](https://user-images.githubusercontent.com/87464956/204961638-8c4381c3-d972-4251-a034-625ab9e4eb17.png)

XGBoost는 간단히 말하자면, **기존에 존재하던 알고리즘을 극한까지 효율성을 끌어올린 모델**이라고 보시면 되겠습니다. 기존 GBM에서 단일 머신을 이용하여 더 빠르고 효율적으로 학습할 수 있게 되었고, Boosting Tree에 대하여 Scalable한 시스템을 가지고 있습니다.

이때, XGBoost가 가지는 특징은 다음으로 정리할 수 있습니다. 이러한 특성들을 기반으로 성능을 끌어올릴 수 있었습니다.

- S**plit Finding Algorithm**

![image](https://user-images.githubusercontent.com/87464956/204961665-95ff5188-8979-4dfd-b01d-d81a7f907b97.png)

XGBoost는 Tree 기반의 Boosting 방법론인 만큼, **Tree의 Split point를 찾는 것**이 매우 중요합니다. 

먼저 예시를 들기 위하여 좌측부터 우측까지 변수 값이 오름차순으로 정렬되어 있다고 가정한 후에, 보통은 모든 점을 자르는 것이 원래 방식이었습니다. 그러나 너무 비효율적인 방식이므로, 다음과 같이 Feature 분포의 Percentiles에 따라 후보 분할 지점을 만들게 됩니다. 이렇게 나눠 줌으로써, 정확도는 조금 떨어질 수 있지만 이제 데이터를 병렬적으로 처리할 수 있어 효율성이 높아지게 됩니다. 

![image](https://user-images.githubusercontent.com/87464956/204961679-e5d9a7dc-02ee-4592-b3cf-e851b7a3a510.png)

이후, 이렇게 나눈 부분을 기점으로 또 자르게 됩니다. 이러한 방식으로 Split point를 찾음으로써, 효율성을 높일 수 있게 되었습니다. 그러나, 너무 잘게 자르게 된다면 Local optimal에 빠질 수 있고, 너무 크게 자르게 된다면 정확도가 떨어질 우려가 있습니다.

- **Sparsity-Aware Split Finding**

![image](https://user-images.githubusercontent.com/87464956/204961694-e4fe91d7-c50d-40af-b054-4d6081ba1e2c.png)

많은 분들이 알다시피, 현실의 데이터에는 입력 데이터의 밀도가 낮은 경우가 빈번합니다. 이때, 밀도가 낮다는 의미는 데이터에 결측치가 많거나, 0의 값이 매우 높거나, 1-hot encoding 등의 방법이 수행된 경우를 뜻합니다. 그렇다면 이러한 방법은 어떻게 해결할 수 있을까요? 정답은 **Feature 분포의 기본 방향을 설정하여, 0과 같은 값들을 분기의 한쪽 방향으로 몰아버리는 것** 입니다.

다음과 같이 value가 없는 데이터를 오른쪽으로 민 후, value가 있지만 class가 0인 데이터들을 또 왼쪽으로 밀어주며 최종적으로 다음과 같은 형태의 Feature 분포를 얻을 수 있게 됩니다.

- **Regularized Learning for avoiding overfitting**

![image](https://user-images.githubusercontent.com/87464956/204961701-9ea2309a-8d12-426c-898a-c6754c687215.png)

XGBoost는 Boosting 계열 모델로, 여러 Tree로부터 받은 값들을 합친 것으로 최종 예측 값을 결정합니다. 이때, Gradient Boosting에서는 Tree들을 학습시키기 위하여 MSE loss function을 사용하는데, **XGBoost는 Regularized Term을 넣어서 정규화 시켜주면서 Overfitting을 피하고** 있습니다.

- **Shrinkage and Column Subsampling**

**Shrinkage:** Boosting tree의 각 단계 이후 마다, 새롭게 추가되는 가중치 $η$를 통하여 Scaling 해 줍니다. 이를 통하여 개별 트리의 영향을 감소하고, 새롭게 추가되는 Tree들도 만들어지는 Strong에 역할을 할 수 있게 합니다.

**Column Subsampling:** 모든 Feature를 사용하는 것이 아닌, 일부 Feature만을 사용하여 다양성을 부여하고 Overfitting을 방지합니다. 

- **Computer System**
    - Cache-aware Access
    - Column Block for parallel learning
    - Out of core computation

해당 방법들은 **모두 컴퓨터 내부에서 효율성을 챙기고자 하는 방법론들**입니다.

## 4. LightGBM

![image](https://user-images.githubusercontent.com/87464956/204961745-5f23e3e5-3f5a-4644-8b12-c54e48b50ddf.png)

LightGBM은 XGBoost와 비슷하게 기존의 Gradient Boosting tree의 문제점을 해결하고자 한 모델입니다. 

먼저 XGBoost에서는 효율성과 Scalability를 향상 시키고자 하였지만 변수들이 많으면서, 데이터의 크기가 클 수록 각 Feature가 모든 분할 지점의 Information Gain을 추정하고자 모든 데이터 Instance를 탐색하게 되어 많은 시간을 소비하여 여전히 효율성과 Scalability 문제가 발생하였습니다.

이때, Light GBM에서는 이를 해결하기 위하여 **두 가지 새로운 기법**들을 제시합니다.

- **Gradient-based One-Side Sampling (GOSS)**

훈련 데이터의 크기가 클수록 각 Feature가 모든 분할 지점의 Information Gain을 추정하고, 이 과정에서 효율성이 떨어지게 됩니다. 그렇다면, 훈련 데이터의 크기를 줄일 수는 없을까요? 

**훈련 데이터의 크기를 줄이기 위한 가장 쉬운 접근 방법은 바로 데이터를 Donw Sampling하는 것입니다.** 

![image](https://user-images.githubusercontent.com/87464956/204961760-e1aba1ee-a026-4df3-a1f7-e60beb8f2d18.png)

LightGBM에서는 서로 다른 Gradient를 가진 데이터가 Information Gain 계산에서 서로 다른 역할을 수행하게 됩니다. 이에, 다음과 같이 **큰 Residual의 절대값을 가지는 데이터가 Information Gain을 계산함에 있어서 더 많이 기여**한다고 가정하여 상위의 큰 Gradient를 가지는 데이터들은 100%로 Sampling하고, 하위의 작은 Gradient를 가지는 데이터들은 무작위로 Sampling하자는 방법입니다.

결과적으로, **해당 방법을 통하여 초기 데이터 분포의 큰 변경 없이 학습**할 수 있게 됩니다.

- **Exclusive Feature Bundling (EFB)**

Feature들이 많을 수록, 데이터들은 매우 Sparse하게 됩니다. 또한 Feature 공간에서 많은 Feature들은 상호 배타적인 관계를 가지는데, 이러한 **Feature들을 안전하게 묶을 수 있는 최적 Bundle 문제를 Greedy algorithm을 통하여 묶어서 하나의 새로운 Feature를 만들고자 하는 방법입니다.**

![image](https://user-images.githubusercontent.com/87464956/204961785-4f3922bf-dc89-4604-b9e5-b17e0177d9e2.png)

먼저, 저희에게 왼쪽의 데이터가 있을 경우, 오른쪽의 Feature map처럼 그릴 수 있게 됩니다. 이후 저희는 Cut-off 값을 정하여서 서로 크게 상관이 없는 Feature들은 하나로 묶어주게 됩니다. 

![image](https://user-images.githubusercontent.com/87464956/204961798-becf040c-1540-4c2d-80b1-ad9a7948246e.png)

해당 예시에서는 Cut-off 값을 0.2로 설정하여서, 2보다 큰 값들은 다음과 같이 잘라주게 됩니다. 최종적으로 저희는 Data table을 다음과 같이 다시 그릴 수 있게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/204961816-9e80f151-fcdf-43e5-aa2c-94927f2245da.png)

## 5. CatBoost

![image](https://user-images.githubusercontent.com/87464956/204961838-758b5304-2ff0-42f1-a5cf-6ba363b62024.png)

마지막으로 Catboost입니다. **Catboost는 Category와 Boosting의 합성어**로, Gradient boosting 기법을 기반으로 하여 Categorical feature의 전처리에 특별한 방법을 사용하는 알고리즘입니다.

이때, 기존 Gradient Boosting 기법의 문제를 해결하기 위하여 Catboost에서는 다음의 두 가지 방법을 도입합니다.

- **Ordered Target Statistics**

![image](https://user-images.githubusercontent.com/87464956/204961856-dc368227-fea6-42fa-9660-f4e8f1f831bd.png)

다음과 같이, **Categorical한 Feature들을 모두 Target에 대한 통계량인 Target statistics로 대체하여 Numerical하게 표현하는 방법을 Target Leakage**라고 합니다. 이는 기존 One-hot encoding 방법보다는 훨씬 개선된 방법이지만, 해당 방법에는 심각한 오류가 한 가지 존재합니다. 

**사실 우리가 하고자 하는 것은 x로 y를 추론하는 것인데, y를 가지고 x를 추론해버리는 상황이 발생한 것입니다.**

이에 Catboost에서는, 이를 **Ordered Target Statistics**를 통해 해결하고 있습니다.

![image](https://user-images.githubusercontent.com/87464956/204961870-960178ed-bc1e-45a2-8169-167b115d2fa5.png)

**Oredered Target Statistics**는, 각 Instance들의 TS값을 현 시점에서 관측되는 history만을 통하여 계산하게 됩니다. 이때, 해당 순서를 맞춰주기 위하여 Arifitial time을 의미하는 $σ$를 제시합니다. 이를 통하여 각 Step마다 각기 다른 **σ** 를 생성해서 이용하여, 현 시점의 관측값만을 이용할 수 있습니다.

- **Ordered boosting**

![image](https://user-images.githubusercontent.com/87464956/204961894-04525e87-924f-4484-adba-2f4d4493b384.png)

두 번째로 해당 논문에서 제시하는 것은 **Prediction shift 문제**를 해결하기 위한 **Ordered boosting**입니다. Catboost는 기본적으로 Boosting 계열의 모델이므로 이전의 모델을 바탕으로 현재의 모델이 만들어지게 됩니다. 그러나, 이전에 이미 만들어진 모델을 만들 때 사용되었던 데이터를 이용하여 또 Gradient를 구하고, 이를 또 근사하는 새로운 모델을 학습한다는 것입니다. 이렇게 **각 Step에서 사용하는 Gradient들은 현재 모델을 만드는 데 사용되는 데이터의 Target값을 사용하기에 연쇄적으로 Prediction shift가 발생**하게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/204961909-d0c950d6-7cfe-4fef-b8d6-4e8af66a5929.png)

Catboost에서는 이러한 Prediction shift 문제를 해결하기 위하여, **Ordered Boosting**을 제안합니다. 이는 다음과 같이 **Training에 사용한 instance에 따라 모델의 집합을 다르게 생성하여 유지**하면, 각 단계에서 독립적인 데이터셋을 사용할 수 있게 된다는 것입니다. 이때, 이를 위하여 Ordered TS처럼 Artifitiial Time인 $σ$를 이용하게 됩니다.

## 6. Tutorial
