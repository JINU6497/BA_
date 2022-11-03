# 목차

1. SVM
    - Kernel
    - Tutorial
2. SVR
    - Tutorial




# 1. SVM

먼저 **SVM**에 알아보기 전에, **Shatter**와 **VC dimension**이란 개념을 알아보도록 하겠습니다.

![image](https://user-images.githubusercontent.com/87464956/199709488-82b719b0-ba58-4f33-b6f0-126c88e2a13c.png)

먼저 **Shatter**란, 다음과 같은 Data들이 존재할 때, 어떠한 함수 f가 이들을 얼마나 분류할 수 있는지를 말하는 능력입니다.

다음의 예시에서 왼쪽의 2차원에서는 4개 이상의 점들은 하나의 함수로는 Shatter가 불가능하며, 오른쪽의 원형 분류기에서는 d개의 차원이 존재할 때 d+1개의 점은 Shatter가 불가능합니다.

이때, 선형 분류기에서는 4개 이상의 점을 Shatter하기 위하여 Kernel method를 사용하게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/199709541-68da31d6-56df-4c04-b685-da5ee4ad17a0.png)

다음으로는 **VC Dimension**입니다. 이는 어떤 함수의 Capacity를 측정하는 지표로, 어떤 함수에 의하여 최대로 Shatter할 수 있는 Points의 수를 의미합니다.  

이 VC Dimension이 **클수록 더 복잡한 모델링**이 가능하지만, **Overfitting 되어서 Generalization 능력이 떨어질 수 있습니다.** 이러한 상황을 **구조적 위험(Structural Risk)**이라고 합니다. 

결론적으로, 이러한 **구조적위험(Structural Risk)**은 데이터의 개수(n)와 VC dimension(h)으로 이루어지는데, 이를 **최소화해야 더 좋은 모델이라고 볼 수 있습니다.** 이때, SVM은 VC Dimension을 최소화 하기 위하여 **Margin**이라는 개념을 사용합니다.

![image](https://user-images.githubusercontent.com/87464956/199709583-097e49ef-b751-42b1-8924-c648101ebbe9.png)

먼저, **SVM**은 다음과 같은 **Binary classification** 알고리즘으로 **복잡도가 매우 낮습니다.** 

![image](https://user-images.githubusercontent.com/87464956/199709618-1f6defc8-d336-4a0d-b15c-453fb2475d44.png)

이때 저희는 **Classification**을 하기 위하여 선형 결정면을 찾는데, 나올 수 있는 결정면은 수도 없이 많습니다. 그렇다면, 어떠한 결정면이 나오는 분류기를 선택해야 더 좋은 걸까요? **바로 Margin이 최대화 되는 분류기를 더 좋은 분류기라고 말할 수 있을 것입니다.**

오른쪽의 예시를 보면 B보다는 A에서 Margin이 커지므로, A를 더 좋은 분류기라고 말할 수 있습니다. 

![image](https://user-images.githubusercontent.com/87464956/199709652-1a773c16-2cc1-45f0-9584-258c87974c7f.png)

이때, **SVM**은 $**y = wx+b$ 로 표현되는 선형회귀 모델이 Margin을 최대로 갖도록 하게하는 것입니다.**

오른쪽의 예시에서는 Binary classification을 하기 위하여 $wx+b>1$이면 $y$를 1로, $wx+b<-1$이면 $y$를 -1로 labeling합니다. 

![image](https://user-images.githubusercontent.com/87464956/199709682-0e0436d2-788c-4a6a-be2c-c4fdc238aa3e.png)

먼저, **SVM**의 **목적함수**와 **제약식**은 다음과 같습니다. 이 제약식의 의미는 $wx_{i}+b=0$의 선이 존재하고 margin의 크기가 1이라고 간주하였을 때, $wx+b=-1 (j_{j}=-1)$ 선의 아래의 값들은 $y_{j}(wx_{j}+b)≥+1$이 되고, $wx+b=1 (j_{j}=1)$ 선 위의 값들 역시 $y_{j}(wx_{j}+b)≥+1$이 되어버리기 때문입니다. 

이후 **제약조건**을 최적화 식에 더함으로써 **Primal Lagrangian Problem**을 만들고, 우리가 구해야 하는 **미지수인 w, b**에 대하여 **편미분**을 한 후에 이를 통하여 **Dual Lagrangian Problme**을 만들 수 있습니다.

**Dual Lagrangian Problme**은 편미분한 **KKT Condition**을 바탕으로 Primal Lagrangian Problem에 w를 대입하고 Lagrangian Multiplier에 대해 Maximization 함으로써 만들어 집니다. **이는 α에 대한 2차식이므로, Convex optimization을 통하여 Optimal한 값을 찾을 수 있습니다.**

![image](https://user-images.githubusercontent.com/87464956/199709728-b8e23369-32c3-4f11-8ac1-d08ea8482747.png)

이때, **Training** **시에는 왼쪽 그림과 같이 분류 경계면이 Margin을 고려하여 학습**하고, **Prediction** **시에는 아래와 같이 중앙의 분류선을 기준으로 Sign function을 사용하여 Classification 합니다.**

이러한 과정을 통하여 모델의 **Training**과 **Prediction**을 진행합니다. 이때, 우리는 **SVM의 핵심**이라고 할 수 있는 **Support Vector**들에 대하여 알아야 합니다.

![image](https://user-images.githubusercontent.com/87464956/199709817-f3200b3b-7f70-453c-9dc6-01f6af3f4095.png)

**SVM의 가장 큰 특징**은, Support Vector 들의 정보만 가지고 있으면 Model을 유지 및 저장 할 수 있다는 것입니다.

**KKT condition**에 의하여 $a_{i}(y_{i}(w^Tx_{i}+b)-1) = 0$의 수식이 만족되는데, 이때 $a_{i}$가 0이라면 수식이 0이 되어버리고, $a_{i}$가 0이 아니라면 $(y_{i}(w^Tx_{i}+b)-1) = 0$이 됩니다. 이때, **해당 수식은 Margin 위에 존재하는 Vector들만을 의미**하고, 이것들이 바로 **Support Vector**가 됩니다.

$x, y$는 데이터로부터, $w, a$ **KKT condition**에 의하여 따로 구할 수 있고, $b$는 위의 **margin 조건식**을 통하여 구할 수 있습니다. 

그렇다면, **Margin의 크기**는 어떻게 구해지는 것일까요? 이는 **Largrangian multiplier $α$**를 통하여 구해집니다.

![image](https://user-images.githubusercontent.com/87464956/199709842-b9660660-2740-4c33-ab8d-819f28e93844.png)

이때, Support Vector는 $y$상에 존재한다는 것을 기반으로 한다면 b는 다음의 식을 만족합니다.

![image](https://user-images.githubusercontent.com/87464956/199709878-ebc38f8e-6108-4ed2-90ac-158d7a499f88.png)

이후, 위의 수식에 $a_{i}y_{i}$를 곱한다면, 다음과 같이 전개가 가능해집니다.

![image](https://user-images.githubusercontent.com/87464956/199709894-fb761679-4df9-4842-baff-80527bce1891.png)

이때, $y$의 값은 1이 아니면 -1이므로, 제곱은 1이 되고, KKT condition에 따라 식을 전개하면 최종적으로 다음과 같은 식을 얻을 수 있습니다.

![image](https://user-images.githubusercontent.com/87464956/199709918-6e5b5f38-ad12-45ca-ab20-a8cbc4b1c0c1.png)

지금까지 배운 **SVM**을 **Hard-SVM**이라고 하는데, **Soft-SVM**이 존재합니다. 이는 **잘못 분류된 Case를 어느 정도 용인하여 Panelty**를 줍니다. 이때, **Hard-SVM** 보다 **오히려 Margin이 커질 수 있고, Noise를 고려하여 더욱 Generalization 될 수 있습니다.**

![image](https://user-images.githubusercontent.com/87464956/199709943-b30d071d-6e16-4c6d-8ee6-ad6b413187eb.png)

Soft-SVM은 다음의 수식들을 통하여 Hard-SVM 과 똑같이 계산될 수 있고, 최종적으로는

![image](https://user-images.githubusercontent.com/87464956/199709968-8ee24d4e-4c7e-4a4a-a7bc-0e539516c818.png)

이를 통하여 **Training** 시에는 분류 경계면이 Margin 을 고려하여 학습되고, **Prediction** 시에는 중앙의 분류선을 기준으로 Sign function으로 Classification하게 됩니다.

이때, **목적 함수에 존재하는 C**는 **Panelty의 허용 영향도를 정하는 Hyper parameter**로, 이것에 따라 Margin의 크기가 정해집니다. 

- C가 커지면 $ξ$ Panelty 허용이 작아지고, 이에 따라서 Margin이 작아집니다.
- C가 작아지면 $ξ$ Panelty 허용이 커지고, 이에 따라서 Margin이 커집니다.

# 2. Kernel

저희는 지금까지의 과정을 통하여 SVM이 무엇인지 알 수 있었습니다. 그러나, **결국 SVM은 선형 분류기**입니다. 그렇다면 선형으로 분류가 불가능한 상황이라면 SVM은 쓸모가 없는 것일까요? **한 가지 아이디어**를 더한다면, 이를 사용할 수 있습니다 !

![image](https://user-images.githubusercontent.com/87464956/199710003-e1136458-c54a-4d11-b4b8-2bb0fd0f86e8.png)

바로 다음과 같이, **데이터를 더 높은 차원으로 매핑 후**에 Decision surface를 통하여 선형 분류를 하는 것입니다.

![image](https://user-images.githubusercontent.com/87464956/199710025-4dfd627a-1bd0-4924-ab28-d9aa2a52d953.png)

이는 다음과 같은 **$Φ$함수를 통하여 mapping** 할 수 있습니다. 이때, q는 p보다 큰 숫자로, 더 큰 차원을 의미합니다. 그렇다면, 이 $Φ$함수는 어떻게 적용되는 것일까요?

![image](https://user-images.githubusercontent.com/87464956/199710050-f98e6ec5-517e-4e9a-901b-4b905da7f747.png)

$Φ$함수의 사용법에 대한 예시를 들어보기 위하여 Soft-SVM Dual Lagrangian Problem을 가져왔습니다. 이때, $Φ$함수를 적용하면 다음과 같은 식을 얻을 수 있습니다.

그런데, 결과값을 본다면 $x$값을 변형 후에 사용하는 것은 결국 **$Φ(x)$의 내적**입니다. 그렇다면, 굳이 $Φ$를 사용하지 말고 직접 내적 값에 해당하는 $Φ(x_{i})^TΦ(x_{j})$만 정의해도 똑같은 효과를 낼 수 있지 않을까요?

이에, 기존 $x$에서 $Φ$를 통하여 고차원으로 매핑시킨 후,  내적하여 $Φ(x_{i})^TΦ(x_{j})$값을 얻는 것에서 $x$에서 바로 $<Φ(x_{i})^TΦ(x_{j})>$를 거쳐버리는 **Kernel 함수**를 이용하자는 것입니다.

이때, Kernel 함수로는 다양한 함수가 존재하는데, 다음과 같은 함수들이 존재합니다.

![image](https://user-images.githubusercontent.com/87464956/199710076-05c8d92a-3114-4a69-971b-e60dfed4d223.png)
