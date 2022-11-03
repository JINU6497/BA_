# 목차

1. SVM
    - 1-1. Kernel
    - 1-2. Tutorial
2. SVR


# 1. SVM

먼저 **SVM**에 알아보기 전에, **Shatter**와 **VC dimension**이란 개념을 알아보도록 하겠습니다.

![image](https://user-images.githubusercontent.com/87464956/199709488-82b719b0-ba58-4f33-b6f0-126c88e2a13c.png)

먼저 **Shatter**란, 다음과 같은 Data들이 존재할 때, 어떠한 함수 f가 이들을 얼마나 분류할 수 있는지를 말하는 능력입니다.

다음의 예시에서 왼쪽의 2차원에서는 4개 이상의 점들은 하나의 함수로는 Shatter가 불가능하며, 오른쪽의 원형 분류기에서는 d개의 차원이 존재할 때 d+1개의 점은 Shatter가 불가능합니다.

이때, 선형 분류기에서는 4개 이상의 점을 Shatter하기 위하여 Kernel method를 사용하게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/199709541-68da31d6-56df-4c04-b685-da5ee4ad17a0.png)

다음으로는 **VC Dimension**입니다. 이는 어떤 함수의 Capacity를 측정하는 지표로, 어떤 함수에 의하여 최대로 Shatter할 수 있는 Points의 수를 의미합니다.  

이 VC Dimension이 **클수록 더 복잡한 모델링**이 가능하지만, **Overfitting 되어서 Generalization 능력이 떨어질 수 있습니다.** 이러한 상황을 **구조적 위험(Structural Risk)** 이라고 합니다. 

결론적으로, 이러한 **구조적위험(Structural Risk)** 은 데이터의 개수(n)와 VC dimension(h)으로 이루어지는데, 이를 **최소화해야 더 좋은 모델이라고 볼 수 있습니다.** 이때, SVM은 VC Dimension을 최소화 하기 위하여 **Margin**이라는 개념을 사용합니다.

![image](https://user-images.githubusercontent.com/87464956/199709583-097e49ef-b751-42b1-8924-c648101ebbe9.png)

먼저, **SVM**은 다음과 같은 **Binary classification** 알고리즘으로 **복잡도가 매우 낮습니다.** 

![image](https://user-images.githubusercontent.com/87464956/199709618-1f6defc8-d336-4a0d-b15c-453fb2475d44.png)

이때 저희는 **Classification**을 하기 위하여 선형 결정면을 찾는데, 나올 수 있는 결정면은 수도 없이 많습니다. 그렇다면, 어떠한 결정면이 나오는 분류기를 선택해야 더 좋은 걸까요? **바로 Margin이 최대화 되는 분류기를 더 좋은 분류기라고 말할 수 있을 것입니다.**

오른쪽의 예시를 보면 B보다는 A에서 Margin이 커지므로, A를 더 좋은 분류기라고 말할 수 있습니다. 

![image](https://user-images.githubusercontent.com/87464956/199709652-1a773c16-2cc1-45f0-9584-258c87974c7f.png)

이때, **SVM**은 **$y = wx+b$ 로 표현되는 선형회귀 모델이 Margin을 최대로 갖도록 하게하는 것입니다.**

오른쪽의 예시에서는 Binary classification을 하기 위하여 $wx+b>1$이면 $y$를 1로, $wx+b<-1$이면 $y$를 -1로 labeling합니다. 

![image](https://user-images.githubusercontent.com/87464956/199709682-0e0436d2-788c-4a6a-be2c-c4fdc238aa3e.png)

먼저, **SVM**의 **목적함수**와 **제약식**은 다음과 같습니다. 이 제약식의 의미는 $wx_{i}+b=0$ 의 선이 존재하고 margin의 크기가 1이라고 간주하였을 때, $wx+b=-1 (j_{j}=-1)$ 선의 아래의 값들은 $y_{j}(wx_{j}+b)≥+1$이 되고, $wx+b=1 (j_{j}=1)$ 선 위의 값들 역시 $y_{j}(wx_{j}+b)≥+1$이 되어버리기 때문입니다. 

이후 **제약조건**을 최적화 식에 더함으로써 **Primal Lagrangian Problem**을 만들고, 우리가 구해야 하는 **미지수인 w, b**에 대하여 **편미분**을 한 후에 이를 통하여 **Dual Lagrangian Problme**을 만들 수 있습니다.

**Dual Lagrangian Problme**은 편미분한 **KKT Condition**을 바탕으로 Primal Lagrangian Problem에 w를 대입하고 Lagrangian Multiplier에 대해 Maximization 함으로써 만들어 집니다. **이는 α에 대한 2차식이므로, Convex optimization을 통하여 Optimal한 값을 찾을 수 있습니다.**

![image](https://user-images.githubusercontent.com/87464956/199709728-b8e23369-32c3-4f11-8ac1-d08ea8482747.png)

이때, **Training** **시에는 왼쪽 그림과 같이 분류 경계면이 Margin을 고려하여 학습**하고, **Prediction** **시에는 아래와 같이 중앙의 분류선을 기준으로 Sign function을 사용하여 Classification 합니다.**

이러한 과정을 통하여 모델의 **Training**과 **Prediction**을 진행합니다. 이때, 우리는 **SVM의 핵심**이라고 할 수 있는 **Support Vector**들에 대하여 알아야 합니다.

![image](https://user-images.githubusercontent.com/87464956/199709817-f3200b3b-7f70-453c-9dc6-01f6af3f4095.png)

**SVM의 가장 큰 특징**은, Support Vector 들의 정보만 가지고 있으면 Model을 유지 및 저장 할 수 있다는 것입니다.

**KKT condition**에 의하여 $a_{i}(y_{i}(w^Tx_{i}+b)-1) = 0$의 수식이 만족되는데, 이때 $a_{i}$가 0이라면 수식이 0이 되어버리고, $a_{i}$가 0이 아니라면 $(y_{i}(w^Tx_{i}+b)-1) = 0$이 됩니다. 이때, **해당 수식은 Margin 위에 존재하는 Vector들만을 의미**하고, 이것들이 바로 **Support Vector**가 됩니다.

$x, y$는 데이터로부터, $w, a$ **KKT condition**에 의하여 따로 구할 수 있고, $b$는 위의 **margin 조건식**을 통하여 구할 수 있습니다. 

그렇다면, **Margin의 크기**는 어떻게 구해지는 것일까요? 이는 **Largrangian multiplier $α$** 를 통하여 구해집니다.

![image](https://user-images.githubusercontent.com/87464956/199709842-b9660660-2740-4c33-ab8d-819f28e93844.png)

이때, Support Vector는 $y$상에 존재한다는 것을 기반으로 한다면 b는 다음의 식을 만족합니다.

![image](https://user-images.githubusercontent.com/87464956/199709878-ebc38f8e-6108-4ed2-90ac-158d7a499f88.png)

이후, 위의 수식에 $a_{i}y_{i}$를 곱한다면, 다음과 같이 전개가 가능해집니다.

![image](https://user-images.githubusercontent.com/87464956/199709894-fb761679-4df9-4842-baff-80527bce1891.png)

이때, $y$의 값은 1이 아니면 -1이므로, 제곱은 1이 되고, KKT condition에 따라 식을 전개하면 최종적으로 다음과 같은 식을 얻을 수 있습니다.

![image](https://user-images.githubusercontent.com/87464956/199709918-6e5b5f38-ad12-45ca-ab20-a8cbc4b1c0c1.png)

지금까지 배운 **SVM**을 **Hard-SVM**이라고 하는데, **Soft-SVM**이 존재합니다. 이는 **잘못 분류된 Case를 어느 정도 용인하여 Panelty**를 줍니다. 이때, **Hard-SVM** 보다 **오히려 Margin이 커질 수 있고, Noise를 고려하여 더욱 Generalization 될 수 있습니다.**

![image](https://user-images.githubusercontent.com/87464956/199709943-b30d071d-6e16-4c6d-8ee6-ad6b413187eb.png)

최종적으로 Soft-SVM은 다음의 수식들을 통하여 Hard-SVM 과 똑같이 계산될 수 있습니다.

![image](https://user-images.githubusercontent.com/87464956/199709968-8ee24d4e-4c7e-4a4a-a7bc-0e539516c818.png)

이를 통하여 **Training** 시에는 분류 경계면이 Margin 을 고려하여 학습되고, **Prediction** 시에는 중앙의 분류선을 기준으로 Sign function으로 Classification하게 됩니다.

이때, **목적 함수에 존재하는 C**는 **Panelty의 허용 영향도를 정하는 Hyper parameter**로, 이것에 따라 Margin의 크기가 정해집니다. 

- C가 커지면 $ξ$ Panelty 허용이 작아지고, 이에 따라서 Margin이 작아집니다.
- C가 작아지면 $ξ$ Panelty 허용이 커지고, 이에 따라서 Margin이 커집니다.

## 1-1. Kernel

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

## 1-2. Tutorial

파이썬을 통하여 **SVM**의 Tutorial을 진행합니다.

제가 이번 Tutorial에서 해보고자 하는 것은 데이터를 load하여 SVM을 통하여 선형 분류를 해본 후, 시각화를 통하여 각 **Hyper parameter**들의 변화가 SVM에 어떠한 영향을 끼치는 지를 알아보고자 합니다. 

마지막으로, 각 알고리즘 별로 최적의 **Hyper parameter** 조합을 찾아주는 알고리즘인 **Grid search** 를 통하여 최적의 **Hyper parameter** 조합을 찾는 것으로 Tutorial을 마치도록 하겠습니다.

각 Hyper parameter들의 종류와 설정값은 다음과 같습니다.

- **kernel :** SVM에 적용할 kernel의 종류
    - ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    
- **C :** 오류를 얼마나 규제할 것인지에 대한 Hyper parameter.  작을수록 Soft margin, 클수록 Hard margin을 의미한다.
    - 1, 50, 100, 150, 200
    
- **degree :** 다항식 kernel 차수.
    - 3, 5, 7, 10

- **coef0 :** 다항식 kernel에 존재하는 상수항 r의 값. ‘poly’, ‘sigmoid’ 일 경우에만 중요.
    - 0, 1, 50, 70, 100

- **gamma :** 결정경계를 얼마나 유연하게 그릴지를 결정하는 것. 클수록 overfitting 발생할 가능성 높아진다.
    - 0.1, 1, 5

```python
"""1. 필요한 Module import"""

import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_wine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
```

이후, 사용할 데이터를 불러옵니다. 이번에 사용할 데이터는 Sckit-learn에서 제공하는 classification data 중 하나인 **wine 데이터**를 사용하도록 하겠습니다.

이는 다음과 같이 **총 3개의 class를 가지고, 각 class 마다 59, 71, 48개의 샘플 데이터를 가지고 있으며, 13개의 변수**를 가지고 있습니다.

![image](https://user-images.githubusercontent.com/87464956/199727939-16b353d4-62c3-42b5-b611-64a16696f1a6.png)

```python
"""2. 데이터 불러오기"""

wine = load_wine()
wine_data = wine.data
wine_feature = wine.feature_names
df_wine = pd.DataFrame(wine_data, columns = wine_feature)
df_wine['target'] = wine.target 

# 각 변수별 Correlation 확인하여, target과 가장 correlation이 높은 두 가지의 변수만으로 선택

train_corr = df_wine.corr()
sns.set(rc = {'figure.figsize':(15,10)}, font_scale = 1.0)
ax = sns.heatmap(train_corr, annot=True, annot_kws=dict(color='r'), cmap='Greys')

# 이때, od280/od315_of_diluted_wines : 희석 와인의 OD280/OD315 비율과 flavanoids : 플라보노이드 폴리페놀이 가장 높게 나왔으므로, 
# 이 두가지의 변수로 SVM Classification 진행
```

![image](https://user-images.githubusercontent.com/87464956/199727981-5d062ac7-fd01-44da-8aae-00aa9db4076e.png)

이때, 각 변수들과 Target과의 상관 관계를 확인한 후, **가장 상관 관계가 높은 변수인od280/od315_of_diluted_wines(희석 와인의 OD280/OD315 비율)과 flavanoids(플라보노이드 폴리페놀)**를 통하여 해당 실험을 진행하도록 하겠습니다.

```python
# Train data와 Test data 비율 0.9:0.1 로 맞춰주고 Shuffle

x = df_wine[['od280/od315_of_diluted_wines', 'flavanoids']]
y = df_wine['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)

sns.scatterplot(x = x['od280/od315_of_diluted_wines'], y = x['flavanoids'], hue=y, s=100)
```

![image](https://user-images.githubusercontent.com/87464956/199728038-3c81c5c1-a62c-49e6-bc62-35fc7659b95b.png)

이후 SVM의 Train에 사용할 data와, 성능 평가에 사용할 Test 데이터를 9:1의 비율로 분리합니다.

또한, **x 축을 od280/od315_of_diluted_wines(희석 와인의 OD280/OD315 비율)로, y축을  flavanoids(플라보노이드 폴리페놀)로 하여 Class 분포**를 찍어보았을 때 다음과 같은 분포가 나옴을 확인할 수 있었습니다. 

**이를 통하여 2로 Label되어 있는 데이터는 비교적 따로 떨어져 있지만, 1, 0으로 Label 되어있는 데이터들은 넓게 분포하며 조금씩 겹치는 양상을 보인다는 것을 알 수 있습니다.**

```python
"""3. 모델링"""

# SVM model setting function
def set_model(kernel, x_train, y_train, C=1.0, degree = 3, coef0=0, gamma = 0.1):
    svm_model = svm.SVC(kernel = kernel, C = C, degree = degree, coef0 = coef0, gamma = gamma).fit(x_train, y_train)
    return svm_model

# 결과 Plot function
def plot_model(svm_model, kernel_name, x, y, C=1.0, degree = 3, coef0=0, gamma = 0.1):
    xx = np.linspace(x[x.columns[0]].min()-0.5, x[x.columns[0]].max()+0.5, 30)
    yy = np.linspace(x[x.columns[1]].min()-0.5, x[x.columns[1]].max()+0.5, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    plt.subplot(1,1,1)
    Z = svm_model.predict(xy)
    Z = Z.reshape(XX.shape)
    color_num = 3
    
    plt.contourf(XX, YY, Z,  cmap= plt.cm.get_cmap('plasma', color_num), alpha=0.2)
    plt.scatter(x[x.columns[0]], x[x.columns[1]], c=y, cmap= plt.cm.get_cmap('plasma', color_num))

    plt.xlabel(f'{x.columns[0]}',fontsize = 15)
    plt.ylabel(f'{x.columns[1]}',fontsize = 15)
    plt.xlim(xx[0], xx[-1])
    plt.ylim(yy[0], yy[-1])
    plt.title(f'SVC with {kernel_name} kernel        C={C}    degree={degree}    coef0={coef0}    gamma={gamma}')
    plt.show()

# 성능평가 function
def get_classifier_eval(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'macro')
    recall = recall_score(y_test, y_pred, average = 'macro')
    f1 = f1_score(y_test , y_pred, average = 'macro')
    print(f'Accuracy:{accuracy:.4f}, Precision :{precision:.4f}, Recall:{recall:.4f}, F1-Score:{f1:.4f}')
```

다음과 같이 SVM을 모델링하는 함수, 시각화를 위한 함수, 성능평가를 위한 함수를 코딩할 수 있습니다.

```python
# 하이퍼 파라미터 Dictionary

Params = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
          'C' : [1 , 50, 100, 150, 200],
          'degree' : [3, 5, 7, 10],
          'coef0' : [0, 1, 50, 70, 100],
          'gamma': [0.1, 1, 5]}
```

모델 실험에 사용하게 될 하이퍼 파라미터는 다음과 같이 Dictionary 형태로 정리하여 쉽게 사용하도록 합니다. 이후 실험을 진행합니다.

```python
"""4. 실험 진행"""

"""4-1. kernel 변화에 따른 변화"""

for kernel in Params['kernel']:
    setmodel = set_model(kernel = kernel, x_train = x_train, y_train = y_train)
    plot_model(svm_model = setmodel, kernel_name = kernel, x=x, y=y)
    get_classifier_eval(setmodel, x_train, y_train)
```

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199723478-816b206b-3984-429e-a164-aa377e26d15c.png width = 700"" /></td><td><img alt="" src = https://user-images.githubusercontent.com/87464956/199725079-4135d30e-69df-4ab8-8a38-ccbf0b16c0f3.png width = 700"" /></td>
  <tr>
</table>

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199725697-670b2b51-6f08-4bbf-b3eb-91de0a0717a3.png width = 700"" /></td><td><img alt="" src =https://user-images.githubusercontent.com/87464956/199725285-25619e4d-e0cb-4e5f-8c22-d13d7ede9e4a.png width = 700"" /></td>
  <tr>
</table>

기본적으로 데이터의 수가 부족해서 인지 모든 kernel 변화를 인지 명확하게 알 수는 없었습니다.

**linear kernel**의 경우에는 데이터도 선형으로 분류하기 좋게 분포되어 있기 때문에 **나름 직선의 형태를 보이면서 데이터를 잘 분류**하며, Metric에서도 높은 점수를 받고 있습니다.

**poly, rbf kernel**은 **조금씩 비선형적인 추세**를 보이기는 하지만 데이터들이 **해당 kernel들로는 잘 구분할 수 없도록** 분포되어 있어 알기 힘든 모습을 보였습니다. 특히 poly kernel은 전체적으로 linear kernel 보다도 낮은 점수를 받았습니다.

**Sigmoid kernel** 같은 경우에는 **완전히 잘못 분류**하는 형태를 보였습니다. 정확한 이유는 모르겠으나  Sigmoid는 Binary classification에 사용되기 때문에, 해당 데이터처럼 3개의 class를 분류하는 Task에는 적합하지 않아서 발생하는 일인 것 같았습니다.


```python



"""4-2. C에 따른 변화"""

for i in Params['C']:
    setmodel = set_model(kernel = 'poly', x_train = x_train, y_train = y_train, C=i)
    plot_model(svm_model = setmodel, kernel_name = 'poly', x=x, y=y, C=i)
    get_classifier_eval(setmodel, x_train, y_train)
```

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199725907-33d1d8fc-d9ce-418a-b58b-656e029ec1cc.png width = 700"" /></td><td><img alt="" src = https://user-images.githubusercontent.com/87464956/199725931-49a05fe2-0b06-42c9-a97a-e7340ce5df26.png width = 700"" /></td>
  <tr>
</table>

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199725970-9c5b6712-901c-4a9c-a8ad-8ff75feafd39.png width = 700"" /></td><td><img alt="" src =https://user-images.githubusercontent.com/87464956/199726000-d073b605-3390-4639-a5bd-eedfc37950cd.png width = 700"" /></td>
  <tr>
</table>

**C는 오류를 얼마나 규제할 것 인가에 대한 Hyper Parameter**입니다. 이에 따라서 **C가 커지면서 분류 경계면이 더 타이트**하게 그려지는 것을 확인할 수 있습니다.  

**성능 면에서는 C가 커질수록 점점 전체적인 성능이 좋아지다가 100에서 최고점을 찍고, 다시 하락**하게 되는 것을 확인할 수 있었습니다. 너무 작지도, 크지도 않은 C를 찾도록 하는 것이 좋아 보입니다.

```python
"""4-3. degree에 따른 변화"""

for i in Params['degree']:
    setmodel = set_model(kernel = 'poly', x_train = x_train, y_train = y_train, degree=i)
    plot_model(svm_model = setmodel, kernel_name = 'poly', x=x, y=y, degree=i)
    get_classifier_eval(setmodel, x_train, y_train)
```

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199726346-e63661df-0f4a-4dfb-bd2a-63d95fa2ff16.png width = 700"" /></td><td><img alt="" src = https://user-images.githubusercontent.com/87464956/199726391-a11b15a3-840f-47d3-8751-4b1901d1f6d7.png width = 700"" /></td>
  <tr>
</table>

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199726440-4fe280d3-fe7f-4ba7-897f-d922fa0cb6fb.png width = 700"" /></td><td><img alt="" src =https://user-images.githubusercontent.com/87464956/199726500-7722d06c-a749-45b4-984b-a7efa3132692.png width = 700"" /></td>
  <tr>
</table>

**degree는 다항식의 차수에 관한 Hyper Parameter**로, ‘poly’ kernel일 경우에만 영향을 미치는 Hyper Parameter입니다. **차수가 높아지면서 점점 더 복잡한 함수로 표현이 가능해짐으로 더 타이트한 경계면**을 그릴 수 있게 됩니다. 그러나, 너무 큰 degree를 사용한다면 overfitting에 빠질 위험이 있어 보입니다.  

물론 데이터에 따라 달라지겠지만, **degree는 오히려 낮은 값을 가질 때 더 좋은 성능**을 보였습니다.

```python
"""4-4. coef0에 따른 변화"""

for i in Params['coef0']:
    setmodel = set_model(kernel = 'poly', x_train = x_train, y_train = y_train, coef0=i)
    plot_model(svm_model = setmodel, kernel_name = 'poly', x=x, y=y, coef0=i)
    get_classifier_eval(setmodel, x_train, y_train)
```

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199726898-6bcc77fc-9af6-4418-a50c-beff35322676.png width = 700"" /></td><td><img alt="" src = https://user-images.githubusercontent.com/87464956/199726944-6add1062-b97e-44f9-b7a4-d45124abdb21.png width = 700"" /></td>
  <tr>
</table>

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199726982-54cd25c7-d089-46cd-92b6-7e70725d9c2a.png width = 700"" /></td><td><img alt="" src =https://user-images.githubusercontent.com/87464956/199727046-b0b1dc14-771d-4e44-bff9-b9c37671835b.png width = 700"" /></td>
  <tr>
</table>

**coef0는 다항식 kernel에 존재하는 상수항인 r의 값으로,  ‘poly’, ‘sigmoid’ 일 경우에만 사용되는 Hyper Parameter입니다.** 이를 조정함으로써, **모델이 높은 차수와 낮은 차수에 얼마나 영향을 받을지를 조정**할 수 있습니다. degree를 고정한 채 coef0값을 조정하니 값을 높이자 다음과 같은 경계면을 그렸습니다.

경계면이 다소 이상해 보이나, **coef0값을 높이니 성능 적인 측면에서는 오히려 전체적으로 좋아**진 것을 확인할 수 있습니다.

```python
"""4-5. gamma에 따른 변화"""

for i in Params['gamma']:
    setmodel = set_model(kernel = 'poly', x_train = x_train, y_train = y_train, gamma=i)
    plot_model(svm_model = setmodel, kernel_name = 'poly', x=x, y=y, gamma=i)
    get_classifier_eval(setmodel, x_train, y_train)
```

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199727254-d1dd5287-8695-403e-b5fb-dac86c23a8ff.png width = 700"" /></td><td><img alt="" src = https://user-images.githubusercontent.com/87464956/199727308-4a96132a-5d12-49c4-b745-7ab223acdaad.png width = 700"" /></td>
  <tr>
</table>

<table>
  <tr>
    <td><img alt=""src = https://user-images.githubusercontent.com/87464956/199727375-17136222-cf6d-4b0a-a555-5533b1b5e95e.png width = 470"" /></td><td><img alt="" src = "" /></td>
  <tr>
</table>

**gamma값이 커지면서 결정면이 훨씬 구불구불해지고 복잡해진 것을 확인**할 수 있습니다. 이 역시 그림 상으로는 조금 이상해지지만, **성능은 전체적으로 gamma값이 커짐에 따라 상승**하고 있는 것을 확인할 수 있습니다.

```python
"""5. Grid search를 통하여 각 kernel 별 최적의 Hyper parameter 탐색"""

Grid_Params = {'kernel':['sigmoid'],
          'C' : [1 , 50, 100, 150, 200],
          'degree' : [3, 5, 7, 10],
          'coef0' : [0, 1, 50, 70, 100],
          'gamma': [0.1, 1, 5]}

grid_svm = GridSearchCV(svm.SVC(), Grid_Params ,refit=True, n_jobs=-1 ,verbose=3)
grid_svm.fit(x_train, y_train)

print(grid_svm.best_params_)
print(grid_svm.best_estimator_)
print('Best Score:', grid_svm.best_score_)
grid_predictions = grid_svm.predict(x_test)
get_classifier_eval(grid_svm, x_test, y_test)
```
최종적으로, **Grid search**를 통해 찾은 **각 kernel 별의 Hyper Parameter**는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/87464956/199737007-03573ab1-e50c-4a56-85b6-62569753cf45.png)


# 2. SVR

SVR이 무엇인지 알아보기 전에, 먼저 **Function fitting의 목적**을 생각해보겠습니다.

![image](https://user-images.githubusercontent.com/87464956/199737719-d4b152f8-d179-47ef-befb-adf5a267b8f8.png)

간단히 생각하면, **Function fitting의 목적**은 다음과 같은 두 가지로 나눌 수 있을 것입니다. 

1. **설명 변수들로 종속 변수를 유의미하게 잘 맞출 수 있어야 한다. Loss function을 통하여 Error를 최소화 하자.**
2. **같은 성능을 가지는 함수라면, 최대한 단순함 함수를 만들어야 한다.**

즉, 다음의 예시와 같이 너무 데이터를 잘 표현하는 파란색의 함수보다는, 검은색의 함수가 더 좋을 수도 있다는 것을 의미합니다.

![image](https://user-images.githubusercontent.com/87464956/199737762-2d33b382-5224-40f4-972b-adf3ccdcfcf4.png)

SVR의 목적함수는 다음과 **Lossfunction과 Flatness(편평도)** 로 이루어집니다. 이때 Flatness는 모델의 단순함을 의미한다고 생각하시면 되겠습니다.

저는 SVR 중 가장 basic한 형태를 가지는 ε-SVR을 통하여 SVR을 설명 드리도록 하겠습니다.

여러분들도 알고 있다시피, 현실의 데이터들은 모두 딱딱 떨어지는 데이터가 아니라, 조금씩의 Noise를 가지고 있습니다. 이에 **ε-SVR은 회귀식을 만드는 과정에서 다음과 같이 회귀선 주변에 ±ε 크기의 ε-tube를 만들어서 이 안에 들어오는 데이터들은 맞춘 것으로 간주하자는 것으로, 즉 어느 정도는 Error를 용인**하겠다는 것입니다.

그러나 이때, ε-tube의 밖에 데이터가 존재한다면, 해당 데이터는 **ξ 만큼의 Loss를 선형적**으로 받게 됩니다. 

이는 Ridge Regression의 목적함수와는 조금 다른 형태를 보입니다. Ridge regression의 목적 함수에는 Squared error가 존재하는데, Squared error에서는 회귀선과 데이터가 멀어질수록 loss가 급격히 증가하기 때문에 Outlier 등이 존재한다면 회귀식이 크게 달라질 수 있습니다. 그러나 SVR에서는 **Hinge loss**를 사용하기에, **ε 까지의 Nosie를 용인하기에 더 Robust하며**, **loss가 증가함**에 있어서도 선형적인 형태를 보이기 때문에 **outlier에 대해서 덜 민감**하게 됩니다. 

![image](https://user-images.githubusercontent.com/87464956/199737802-5b8e9e76-ec73-40bd-be00-017c4c9a36fa.png)

먼저, 다음과 같은 선형 회귀식을 추정하게 됩니다. 이때 기본적으로 명심해야 할 부분은 **SVR은 선형 회귀식**이라는 것입니다. 이후 저희는 Kernel trick을 사용함으로써 비선형으로 추정할 수 있게 됩니다.

추정해야 하는 선형 회귀식에서, 저희는 x와 y를 알고 있습니다. 이에 W와 b를 구하는 것을 목표로 하게 됩니다.

SVR의 목적 함수는 아래의 식과 같습니다. 이때, 노란색에 해당하는 부분의 연산이 의미하는 것은 함수의 loss를 줄이자는 것으로, 함수의 정확도를 올리자는 것이며 빨간색의 부분은 Flatness를 의미하는 부분입니다.

![image](https://user-images.githubusercontent.com/87464956/199737832-8443ca1a-69a8-47ee-b0bd-d6c534f66711.png)

목적 함수는 다음의 회색 박스와 같은 제약식을 가집니다. **다음의 제약식에서 설명하는 것은 ε-tube의 위냐 아래냐**에 따라 달라지는 부분을 말합니다. 

첫 번째 식에서는 추정값에서 y를 뺸 경우, 최소한 ε보다는 작거나 같아야 한다는 것을 의미하며, 이를 만족하지 못하면 패널티를 준다는 것을 의미합니다. 즉, 실제값이 함수보다 아래에 있을 때 패널티를 주는 것을 의미합니다.

두 번째 식에서는 실제값이 추정값보다 더 클 때 최소한 ε보다는 크거나 같아야 한다는 것을 의미하며, 이를 만족하지 못하면 패널티를 준다는 것을 의미합니다.

즉, **모든 데이터들은 회귀식의 위에 존재하느냐, 아래에 존재하느냐에 따라서 둘 중 하나의 제약 조건**에 걸리게 되는 것입니다.

![image](https://user-images.githubusercontent.com/87464956/199737863-dc2d3ea2-d36e-4f67-9240-72a8f1e4441f.png)

이후 계산은 SVM과 동일합니다. 주황색의 선으로 표시된 부분은 원래 문제의 제약식을 의미하고, 노란색 부분은 $ξ, ξ^*$가 0 보다는 커야 하므로 이에 대한 부분을 의미합니다. 빨간색 부분에서는 개별적인 객체들에 대해서 말하고 있으며 이렇게 **Primal Lagrangian Problem**이 만들어집니다.

![image](https://user-images.githubusercontent.com/87464956/199737903-7203f9f0-fbf9-480f-b693-bac66daf80d0.png)

이때 저희가 **구해야 하는 미지수는 b, w, ξ** 입니다. 이에 각각에 대하여 편미분을 진행합니다. 

![image](https://user-images.githubusercontent.com/87464956/199737920-cd8cd45b-3737-4c32-a9fe-4922d5c50090.png)

이후 Primal Problem의 최적해 조건을 $L_{P}$에 넣어서 다음과 같이 α에 해당하는 **Dual Lagrangian problem**을 만듭니다. 이렇게 만들어진 α에 대한 2차식을 **Convex optimization**하게 되고, **데이터에 따라서 α는 유니크한 솔루션**을 가지게 됩니다. 이렇게 w값을 구할 수 있게 되면서, **Decision function**은 다음과 같은 회귀식을 가지게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/199737942-16932688-e0ea-4b4e-8f7c-704478064855.png)

이전까지는 선형식이였기 때문에 파란색 부분이 **$x^T_{i}x_{j}$** 로 표현되었지만, **비선형 모델을 만들기 위하여 Kernel trick**을 사용할 경우, 다음과 같은 식으로 표현할 수 있습니다. **Decision function**역시 이에 따라 달라지게 됩니다.
