# **Chapter1. Dimensionality Reduction**

[Code Example](https://github.com/JINU6497/BA_/tree/main/1_Dimensionality%20Reduction)

고려대학교 강필성 교수님의 Business analytics 수업을 기반으로 하는 자료임을 밝힙니다.

보통 머신러닝에서 Business Analytics의 과정은 크게 다음과 같이 이루어집니다.

**1. Pre-Processing: Normalization, Dimension reduction, Image processing, etc**

**2. Learning: supervised, unsupervised, minimization, etc**

**3. Error analysis: precision/recall, overfitting, test/cross validation data, etc**

그렇다면 **Pre-Processing 과정에 존재하는** **Dimension reduction**는 무엇일까요?

$x_{1}$부터 $x_{d}$까지의 데이터가 있는 $X$데이터가 있을 때, 차원의 수인 $d$가 너무 많아서 학습을 잘 못하거나, 가지고 있는 객체의 수보다도 차원이 많아서 통계적인 가정을 만족하지 못하는 알고리즘이 존재합니다. 

이때, 학습을 방해하지 않는 선에서 $x_{1}$부터 $x_{d}$까지의 데이터를 $x_{1}$부터 $x^{'}_{d}$까지 $X$의 차원을 축소시키는 것이 **Dimension reduction입니다.** 



![image](https://user-images.githubusercontent.com/87464956/195574729-4dc88285-1a3f-4efb-a6d2-78e77eca9643.png)

<img src = 'https://user-images.githubusercontent.com/87464956/195574790-ce5c9bfc-93d8-4aa8-8c83-00cc99caec64.png' width = '10%' height = '10%'/>

<img src = 'https://user-images.githubusercontent.com/87464956/195574804-b7334e4e-1248-4c86-89ae-d99ee667882d.png' width = '30%' height = '30%'/>


이러한 식을 만족하여, 모델 효율성을 높여주는 것이 Dimensionality Reudction입니다.

Dimensionality Reudction의 필요성을 **차원의 저주(Curse of dimension)** 측면에서도 설명할 수 있습니다.



![image](https://user-images.githubusercontent.com/87464956/195575160-5a47cea8-da66-406a-8dd9-577dd0fd22af.png)



이렇게 $n$ 차원에서는 동일한 정보를 보존하기 위해서는 $2^n$개의 관측치가 필요합니다.

**차원의 저주(Curse of dimension)** 는 다음과 같이 우리가 동일한 정도의 설명력을 가지기 위해서는 변수가 선형적으로 증가하면 객체는 기하급수적으로 증가한다는 것을 말합니다.

이에 우리는 본질적인 의미를 가지는 **Intrinsic dimension**을 찾아야 하는데, 이는 상대적으로 original dimension보다 작을 수 있습니다.

이러한 **차원의 저주** 문제를 해결하기 위해서는 우리는 **차원 축소 테크닉**을 사용합니다.

먼저,  차원 축소의 방법론은 다음과 같이 크게 2가지로 나눌 수 있습니다.



![image](https://user-images.githubusercontent.com/87464956/195575196-a5ac208e-70c3-40a4-9fb1-67124cbc185d.png)



**supervised Dimensionality reduction**방식은 중간에 알고리즘 또는 모델이 개입을 하는 것 입니다. supervised feature selection을 통하여 차원을 줄이고, 알고리즘에서 feedback roof를 통하여 개선할 수 있습니다.



![image](https://user-images.githubusercontent.com/87464956/195575235-32909440-f8e3-427e-89d0-60afed12ff55.png)



**unsupervised Dimensionality reduction**는 feedback roof가 존재하지 않습니다. 특정한 방법이나 지표를 사용하여 한번만 실행하여 변수를 줄이는 방법입니다. 

다음으로, 차원 축소에 대한 결과물에 따라 **Feature selection** or **Feature extraction 방법**으로 나눌 수 있습니다.



![image](https://user-images.githubusercontent.com/87464956/195575265-d6acaed0-2922-4f3a-81cf-62ce03f9b425.png)



**Feature** S**election**은 말 그대로 현재 존재하는 변수들로부터 **부분집합**을 뽑아내는 것 입니다.



![image](https://user-images.githubusercontent.com/87464956/195575297-3cd9fcb5-f16c-4eda-bdee-a704b1c14e1a.png)



**Feature Extracton**은 기존 변수들의 조합을 통하여 새로운 변수 집합을 만들어 내는 것을 말합니다.

결론적으로 우리의 **목적**은 다음과 같은 방법론들을 통하여 모델에 Best하게 Fit하는 S**ubset of variables**를 찾는 것이라고 볼 수 있고, 이러한 과정을 거치면서 다음과 같은 효과를 얻을 수 있습니다.

- **변수들 간의 상관관계 제거**
- **관리해야 하는 변수들이 단순화 됨으로써, 사후 과정이 단순해짐**
- **중복되거나 불필요한 정보 제거**
- **차원이 줄어듬으로써 시각화가 더 용이해진다.**






