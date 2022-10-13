## **1.** Genetic Algorithm

**Stepwise selection**보다도 시간은 오래 걸리나 **Optimal한 performance 내는 알고리즘**

즉, 계산 시간을 더 투입하여 **Local search(forward/backward/stepwise)**의 성능을 improve시키자는 것이 GA의 목적.

이를 **Meta-Heuristic approach**라고도 하는데, trial or error 방식을 사용하는 과정에서 무작위로 trial하는 것이 아니라 어떻게 더 효율적으로 trial 할 수 있을까를 고안하는 것.

수학적 최적화 알고리즘은 **Natural system**에 영향을 받아서 만들어진 것들이 많다.

이때 **genetic algorithm**은 양성생식에 의한 진화를 모방하는 알고리즘으로 다윈의 자연선택설을 모사하여 수학적인 최적해를 찾는 알고리즘.

 

이때 다음의 세가지 중요한 과정을 거친다.

- **selection**: 현재 객체들 중 상대적으로 우수한 것을 찾자.
- **cross over**: 기존 해들을 가져다가 새로운 대안을 찾자.
- **mutation**: local optima에 빠지면 돌연변이를 만들어 이를 점프하여 새로운 optim을 찾자.

최종적으로, 모델은 다음과 같이 구성되어 있다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/80a63756-46d2-4651-a6ea-8f37ca34b5fa/Untitled.png)
