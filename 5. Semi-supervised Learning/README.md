# 목차

1. Semi-supervised learning 이란?
2. Consistency Regularization
3. Pi-Model
4. Mean Teacher
5. Virtual Adversarial Training(VAT)
6. Unsupervised Data Augmentation(UDA)
7. Tutorial


## 1. Semi-supervised learning 이란?

![image](https://user-images.githubusercontent.com/87464956/209424583-11d54a43-9d12-4a90-b364-63542c87d9bc.png)

먼저, **Semi-supervised learning**이란 **기본적으로는 Supervised learning이지만, 소수의 Labeled Data와 다수의 Unlabeled Data를 함께 사용하는 기법**을 말합니다. 그렇다면, 저희는 왜 이러한 기법을 사용할까요?

저희는 **Label이 있는 Data를 통하여 학습하는 방법을 Supervised learning**이라고 합니다. 이렇게 Label이 있는 Data를 통하여 학습한다면, 분명 모델을 훨씬 더 좋은 방향으로 학습을 진행할 수 있을 것입니다. 그러나 **현실 세계에서는 Label이 있는 Data를 만드는 데 비용이 많이 소모된다는 문제점**이 존재합니다.

그러나 모두 알다시피 네트워크의 구조가 복잡해질수록 Parameter가 많아지면서 학습 시에 더 많은 Data를 필요로 하기 때문에, **양질의 Label 정보를 가지는 데이터를 투입하지 않는다면 해당 네트워크는 충분한 학습을 진행하지 못할 것**입니다.

이에 저희는, **Labeling이라는 정보가 없이도 데이터 자체에서 가지는 정보를 통하여 Labeling을 할 방법**이 필요하게 되었습니다.

![image](https://user-images.githubusercontent.com/87464956/209424589-ce34c442-15b5-4422-b11d-c5415fab6578.png)

이러한 **Semi-supervised Learning이 잘 수행되기 위해서는 Data와 Model들이 몇 가지 Assumption**을 만족해야 합니다. 

1. **Clustering Assumption** 
    
    $X$데이터의 분포, 즉 $P(X)$가 어느정도 $P(y|x)$에 대한 정보를 가져야 한다.
    
2. **Smoothness Assumption**
    
    $X$데이터들끼리 서로 가까이 있을 경우, 그 데이터의 $Y$값도 가까이 있어야 한다.
    
3. **Low Density Assumption**
    
    모델의 Decision Boundary는 입력 $X$값들의 Density가 낮은 곳을 지나가는 것을 선호해야 한다.
    
4. **Manifold Assumption**
    
    원래 공간의 입력 Data는 원래의 특성을 반영하는 낮은 차원의 Manifold(Representation Feature)가 있을 것이다.
    
![image](https://user-images.githubusercontent.com/87464956/209424594-539f7bf4-3055-4145-b0b8-51b747f630d5.png)

이때, **Semi-supervised leanring**은 크게 다음과 같은 갈래로 나누어진다고 볼 수 있습니다. 이번 Tutorial에서는 다음의 **Consistency Regularization Method**를 중점적으로 알아보도록 하겠습니다. 

이는 다음과 같이 **Unlabeled Data에 조금의 Perturbation을 가해주어서 원본 Unlabeled Data에 대한 Predict value과 변화된 Unlabeled Data에 대한 예측 값이 크게 변화하지 않아야 한다는 가정**을 가지고, Consistency하게 예측하도록 만들자라고 말하는 방법론입니다.

## 2. Consistency Regularization

![image](https://user-images.githubusercontent.com/87464956/209424599-da976067-de65-42cd-886e-0fec9577600d.png)

**Consistency Regularization**은 **어떠한 Unlabeled Data에 현실적인 Purterbation을 가해주어서 원본 데이터와 비슷한 어떠한 데이터를 새롭게 생성한다고 할 때, Perturbated Unlabeled data에 대한 Prediction과 원본 Unlabeled Data의 Prediction의 결과는 크게 다르지 않아야 한다**는 것입니다. 

이때, 최종 Output 끼리의 차이를 비교할 수도 있고, Feature Space에서의 차이를 비교할 수도 있는데, 결과적으로 어느 방법이든지 차이가 작아야 한다는 것이 Consistency Regularization의 핵심이라고 보시면 되겠습니다. 

![image](https://user-images.githubusercontent.com/87464956/209424601-730596af-7109-4173-ac21-3fd3e29406cc.png)

해당 Tutorial에서는 Consistency Regularization 방법 중 다음의 4가지 모델을 Cover하도록 하겠습니다.

## 3. Pi-Model

![image](https://user-images.githubusercontent.com/87464956/209424602-a1d1fb48-c3b2-4013-8af4-967ee592a760.png)

먼저 **Π-Model**입니다. 이는 하나의 FFN에 두 번의 Perturbation을 적용하며, Latent vector가 아닌 Output vector들이 Consistent하도록 학습하자고 말하는 방법론입니다. 해당 논문은 독립적으로 제시된 논문은 아니고, 이후에 제시되는 **Temporal Ensemble**의 설명을 위해 사전에 개발된 모델이라고 보시면 되겠습니다. 그렇다면, **Π-Model**을 왜 굳이 다루는 걸까요?

![image](https://user-images.githubusercontent.com/87464956/209424605-db35ac99-9391-409c-bd6c-b2c6ad8b72a0.png)

다음과 같이 **Π-Model**은 매우 단순한 구조를 가지고 있습니다. 그러나, 이는 **향후 나타나는 다양한 방법론들의 기본 구조가 되는 모델**이며, 이전에 제시된 Ladder Network가 Feature space를 기반으로 Consistency Regularization을 한 것과는 달리 Output을 기반으로 Consistency Regularization을 진행하기에 매우 간단한 구조를 가지고 있습니다. 

해당 모델의 구조를 보시면, 단 1개의 Encoder만을 사용하여 Supervised Loss를 구하고, Stochastic Augmentation을 통하여 Unsupervised Loss를 계산하여 Regularization을 주고 있음을 확인할 수 있습니다.

## 4. Mean Teacher

![image](https://user-images.githubusercontent.com/87464956/209424607-e0c1489e-9eca-4cb6-ae44-b6663ea3f508.png)

**Mean Teacher**는 다음과 같이 **Π-Model**과 밀접한 관련이 있는 **Temporal Ensemble** 이후에 나온 모델로, **Temporal Ensemble**이 Reliable Target을 만들기 위하여 Label prediction을 EMA 하여 Ensemble 하는 과정에서, 한 Epoch 당 한 번 Target이 바뀌게 되므로 큰 Dataset을 학습하기에는 힘들다는 점을 보완하고자 하는 모델입니다. 

이를 보완하고자, **Mean Teacher에서는 Label prediction 대신 Model weight를 EMA**합니다.

![image](https://user-images.githubusercontent.com/87464956/209424609-f3f3fc56-3adf-4d48-a8b6-d457f57ba9de.png)

이때, **해당 논문에서는 Unsueprvised Loss를 위한 Target의 Quality를 어떻게 높일 수 있을까** 라는 것에 중점을 두고 있습니다. 이때 해당 논문에서는 **Student model을 복제하는 것 대신 Teacher model을 조심히 선택하자**라는 방법을 취하고 있으며, 이때 **추가적인 Student model training 없이 더 나은 Teacher model 방법**을 제안하고 있습니다.

![image](https://user-images.githubusercontent.com/87464956/209424612-12370a01-bb4c-4de0-bc27-f705ab404bb1.png)

**Mean Teacher**의 모델 구조는 다음과 같습니다. 이때, 간단한 비교를 위하여 Temporal Ensemble의 모델을 같이 가져와 보도록 하겠습니다. 

다음과 같이 Prediction을 EMA하는 Temporal Ensemble과는 다르게 **Mean Teacher에서는 Model의 weight를 EMA**하고 있습니다. 이렇게, Model의 weight를 EMA하는 경우 마지막 Weight를 직접 사용하는 경우보다 더 정확한 모델을 만들 수 있게 되며, Temporal Ensemble은 Epoch마다 EMA Prediction을 업데이트 하지만 **Mean Teacher는 매 Step마다 업데이트가 가능하다는 장점**이 있기에 Mean Teacher를 사용하게 됩니다. 

## 5. Virtual Adversarial Training(VAT)

![image](https://user-images.githubusercontent.com/87464956/209424618-fa743542-231d-4cac-8a9f-0beca46d6a6d.png)

다음으로는, **Π-Model**에서 한번의 FNN에 두 번의 Perturbation을 가해주는 것에서 영감을 얻어서 파생되었다고 볼 수 있는 VAT모델입니다. 이때 **VAT에서는 Random 하게noise를 가해주지 말고, 모델이 가장 취약한 방향으로의 Adversarial noise를 가해주어서 모델의 강건성을 높이자** 라고 말하고  있습니다.

![image](https://user-images.githubusercontent.com/87464956/209424621-404ddde8-1203-4221-bbce-7e08c0d19138.png)

먼저 **VAT**에서는 **기존 Perturbation 기법의 문제점을 다음과 같이 말하고 있습니다.** 왼쪽의 예시처럼, 원본 데이터인 “How are you”와 같은 음성 데이터나, Panda같은 Image data가 존재한다고 생각해보도록 하겠습니다. 이때, 저희가 보기에는 아무런 의미 없어 보이는 Gaussian noise를 다음과 같이 취해주었을 때, 원본 데이터와는 아예 다른 “Open the door”나 Gibbon같은 데이터가 생성되어버리는 문제점이 존재하였습니다.

이를 Embedding space 상에서 생각해본다면, 오른쪽의 예시에서 오른쪽의 빨간색 점처럼 노이즈를 **취약한 방향으로 조금만 노이즈를 주는 순간 실제로 본인이 가지는 Label에서 다른 Label인 파란색으로 넘어갈 수 있다**는 것입니다.

![image](https://user-images.githubusercontent.com/87464956/209424624-c0cf67a4-9129-4273-84e4-4d5e526aee57.png)

이에 **VAT**에서 하고자 하는 것은 초록색이 한 Class의 Basis distribution이라고 가정하고, 보라색을 다른 Class의 basis distribution이라고 할 때, 보라색 Manifold 상에서 좌, 우에 sampling된 데이터들이 모여 있는 것을 확인하실 수 있습니다. 이때, 이 데이터들만 가지고 Adversarial training을 한다면 가운데의 사진처럼 Decision boundary가 형성됩니다. 

그러나 우리는 오른쪽과 같이 좌우로 Adversarial Direction을 설정하여 Training을 하고 싶습니다. **즉, 우리가 원하는 것은 취약한 방향으로 Adversarial Training을 진행하여 취약한 부분을 좀 더 보완**해주고자 하는 것입니다.

![image](https://user-images.githubusercontent.com/87464956/209424626-ebaeba10-341c-4289-bfa1-c453115e4c1f.png)

**VAT**의 전체적인 모델 구조는 다음과 같습니다. 먼저, Label된 데이터와 Unlabeled된 데이터 둘 다에 대해서 Adversarial example을 만들어줍니다. 이는 결국 해당 논문에서 말하는 핵심 파트라고 할 수 있는데 이때 **아무렇게 Perturbation을 진행하지 말고,  Adversarial direction을 찾아서 이에 해당하는 Adversarial example을 만들겠다는 것입니다.**  이후 이를 모델에 집어넣은 후에는 실제 정답이 있는 Label된 데이터에 대해서는 모델이 예측하는 값과 정답과 비슷하도록 **Cross-entropy loss**를 취해주고, 실제 Example과 Adversarial example 사이에서의 모델 output의 분포가 최대한 같도록 **KL Divergence loss**를 사용합니다.

결국 전체적으로는 **Labeled data에 대해서는 정답을 맞추고, Unlabeled data에 대해서는 변형된 대상과 원본 대상의 Distribution을 최대한 같게 한다**라는 Consistency Regularization의 큰 흐름을 따라간다고 볼 수 있겠습니다.

## 6. Unsupervised Data Augmentation(UDA)

![image](https://user-images.githubusercontent.com/87464956/209424631-b01855e2-4218-405c-9f4e-32459a7352a7.png)

**Unsupervised Data Augmentation, 즉 UDA**는 **기본적으로 데이터를 Augmentation 하는 과정에서 남들이 연구해 놓은 좋은 Augmentation 기법을 이용하자라고 말하는 방법론**입니다.

결론적으로, **기존에 Noise를 주어 데이터를 Perturbation하는 것보다, Advanced한 방법을 통하여 Data를 Augmentation 하면 더 좋은 효과**를 볼 수 있다는 것입니다. 이때, **UDA에서는 Vision, NLP 등 다양한 분야를 포함하여 효과적인 Data Augmentation 방법론을 제시**하며, 이와 함께 Consistency loss와 Supervised loss를 통하여 모델을 학습하는 Semi-supervised learning 방법론을 제안하고 있습니다.

![image](https://user-images.githubusercontent.com/87464956/209424635-1c4d54f9-3136-45e6-82df-b2dc3846d75c.png)

**UDA**의 전체적인 구조는 다음과 같습니다. 지금까지 봐왔던 방법론들과 비슷하게, **Label이 있는 데이터에 대해서는 기본적으로 잘 맞춰야 하므로** **Supervised cross-entropy가 Loss로 들어갑니다. 또한 Unsupervised Consistency loss에서는 Unlabeled data가 있을 때, 원본 Raw data와 이를 Task에 맞게 Augmentation한 Data가 똑같은 모델에 들어갈 경우 분포가 서로 비슷하게 하는 역할**을 하게 됩니다. 이 두 가지 Loss들이 더해져서 최종 Loss가 생성됩니다.

![image](https://user-images.githubusercontent.com/87464956/209424639-26b94d62-ac1c-43e7-af3f-ab7a1f451226.png)

**최종적으로 만들어지는 Loss**는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/87464956/209424646-09a4bd11-32c1-453f-b11b-953d4d987463.png)

이전 모델들이 Image data에 초점이 맞춰져 있는 것과는 다르게, 해당 모델에서는 다양한 Task에 맞는 Augmentation 방법론을 제시해주고 있습니다. 가장 먼저 **NLP를 위한 Back-translation**입니다. 다음의 예시를 들어 설명해보면, 먼저 영어 문장을 프랑스어로 번역하는 모델 1을 훈련 시키고, 프랑스어 문장을 영어로 번역하는 모델 2를 따로 훈련 시킵니다. 이후, 저희가 Augmentation하고자 하는 문장이 있을 때 이를 모델 1에 넣어 프랑스어로 만들고, 이렇게 만들어진 Output을 다시 모델 2에 넣어서 영어 문장으로 만들어서 Data를 Augmentation하는 간단한 방법입니다. 이를 통하여, **원본 문장의 의미를 보존하면서 다양한 문장을 얻을 수 있습니다.**

![image](https://user-images.githubusercontent.com/87464956/209424651-9a7450f2-f256-4755-a3f0-840c4e7a8b82.png)

다음으로는 **Word replacing with TF-IDF for Text Classification**입니다. 이전 페이지의 Back-translation이 문장의 Semantic information은 잘 보존할 수 있으나 주제 분류와 같은 Task에는 알맞지 않을 수 있기에, 이러한 경우에는 다음과 같은 방법을 사용합니다. 이는 **문장 내의 몇몇 핵심 단어들이 주제에 대해 Informative할 수 있으므로, Low TF-IDF 점수를 가지는 Uninformative한 단어들을 주제에 대해서 더 많은 Information 가지고 있는 단어로 대체**하는 방법입니다. 다음의 예시와 같이 This라는 단어를 같은 의미는 아니지만 비슷한 의미를 가지면서, 문장을 더 informative하게 만드는 A로 대체하는 것입니다.

![image](https://user-images.githubusercontent.com/87464956/209424656-f711831f-77f0-4801-b30a-f80f38836ba6.png)

마지막으로 **Image classification에 대한 Augmentation 방법**론입니다. 먼저 **Rand Augment**을 알기 전에, **Auto augment**에 대해서 설명 드리겠습니다. 이는 강화학습을 통하여 **Augmentation 하는 방법으로, Python Image Library에서 모든 변환 방법들을 조합하여 최적의 이미지 변환 방법을 찾아서 데이터 증강에 사용하는 방법**입니다. 해당 논문에서 제시하는 **Rand Augment**는 **Auto Augment** 방법을 개조한 방법으로, **최적의 이미지 변환 방법을 찾는 대신에 Uniformly sampling하여 Data를 Augmentation하는 방법**입니다. 이를 통하여 최적의 이미지 변환 방법을 찾을 시간을 사용하지 않아서 빠르게 Augmentation 할 수 있고, labeled data가 필요하지 않게 되었습니다.

## 7. Tutorial

전체적인 실험은 Microsoft 사에서 배포하는 Library인 **“USB: A Unified Semi-supervised learning Benchmark for CV, NLP, and Audio Classification”** 를 통하여 진행하였습니다.

### 7-1. Model 설정

해당 실험에서 진행하고자 하는 것은 **4가지의 Consistency Regularization 모델**들과 함께 **한 가지의 Holistic Method 방법론 MixMatch**를 더하여 총 5가지의 모델에 대하여 실험 결과를 비교하고자 합니다. 이때 사용할 모델들은 다음과 같습니다. 

- Pi-Model
- Mean Teacher
- Virtual Adversarial Training(VAT)
- Unsupervised Data Augmentation(UDA)
- MixMatch

### 7-2. Data

해당 실험에서는 총 3가지의 Dataset을 사용하고자 합니다. 

- **Cifar-100**: 기계 학습에서 흔히 사용되는 이미지 벤치마크 Dataset으로 32 X 32 크기의 60000개의 이미지로 이루어져 있습니다. 이때 100개의 Class로 분류되어 있는데, 각 100개의 Class 이미지는 20개의 Super class로 또 분류됩니다.
- **EuroSAT**: 인공위성 Data로, 27,000개의 Labeled Data로 구성되어 있으며 10개의 Class로 이루어져 있습니다. 이때, Class는 각 토지가 어떻게 이용되는 가를 의미합니다.
- **SVHN**: Google Street View에서 수집된 숫자 Dataset으로, 건물 번호나 표지판과 같은 Real world data로 이루어져 있습니다. 총 10개의 Class의 32 X 32 Image로 이루어져 있습니다.

### 7-3. Variable

해당 실험에서는 각 데이터들에 대한 모델들의 **Label data의 수**와 **Unlabeled data의 비율**을 변경해가며 실험을 진행해보도록 하겠습니다. 이때, Unlabeld ratio가 말하는 것은 labeled data가 64개, Unlabeled ratio가 1인 경우는 Unlabeled data가 64개, Unlabeled ratio가 2인 경우는 Unlabeled data가 128개로 늘어나는 것을 의미합니다.

이때, Data에 대해서는 각 Data의 모든 Class의 수를 그대로 이용했습니다.

- Label data의 수
    - CIFAR-100 = [200, 300, 400]
    - EuroSAT = [20, 30, 40]
    - SVHN = [40, 100, 150]
- Unlabeled ratio = [1, 2, 5]

Labeled data가 늘어난다면, 성능이 좋아질 것이고 Unlabeled ratio가 늘어나면 늘어날수록 성능이 떨어질 것이라는 가설을 바탕으로 실험을 진행해보도록 하겠습니다.

### 7-4. Code

가장 먼저, 라이브러리인 **“USB: A Unified Semi-supervised learning Benchmark for CV, NLP, and Audio Classification”** 를 사용하기 위하여 가상 환경을 만든 후, 해당 라이브러리를 설치해줍니다. 

```python
"""1. 필요한 모듈 import"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
 
import numpy as np
import pandas as pd 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  #for 문의 진행상황 알려줌 
import semilearn
from semilearn import get_dataset, get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
 
# GPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

다음으로는, 실험에서 사용할 Hyper Parameter들을 다음과 같이 Config 파일을 통하여 쉽게 정리해줍니다. 이때, 해당 Congif에서 변화를 줌으로써 쉽게 어떠한 Dataset을 불러올 것인지, 어떤 Model을 사용할 것인지 등을 설정할 수 있습니다.

아래는 예시를 들기 위한 cifar10 data를 fixmatch에 적용한 예시입니다.

```python
"""2. Config 간단히 정리"""

config = {
    'algorithm': 'fixmatch',
    'net': 'vit_tiny_patch2_32',
    'use_pretrain': True, 
    'pretrain_path': 'https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_tiny_patch2_32_mlp_im_1k_32.pth',

    # optimization configs
    'epoch': 1,  # set to 100
    'num_train_iter': 5000,  # set to 102400
    'num_eval_iter': 500,   # set to 1024
    'num_log_iter': 50,    # set to 256
    'optim': 'AdamW',
    'lr': 5e-4,
    'layer_decay': 0.5,
    'batch_size': 16,
    'eval_batch_size': 16,

    # dataset configs
    'dataset': 'cifar10',
    'num_labels': 40,
    'num_classes': 10,
    'img_size': 32,
    'crop_ratio': 0.875,
    'data_dir': './data',
    'ulb_samples_per_class': None,

    # algorithm specific configs
    'hard_label': True,
    'uratio': 2,
    'ulb_loss_ratio': 1.0,

    # device configs
    'gpu': 0,
    'world_size': 1,
    'distributed': False,
    "num_workers": 2,
}
config = get_config(config)
```

이렇게 Config를 설정했다면, 다음과 같이 모델을 정의합니다.

```python
"""3. 모델 정의 """
algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
```

이후, 사용할 Dataset을 정의해줍니다. 이때, 위의 Config에서 정의한 대로 Dataset으로는 Cifar10 사용하고, Label data의 수는 40개, Class는 10개 그대로 사용하였습니다.

```python
"""4. Dataset 정의"""

# 
dataset_dict = get_dataset(config, config.algorithm, config.dataset, config.num_labels, config.num_classes, data_dir=config.data_dir)
train_lb_loader = get_data_loader(config, dataset_dict['train_lb'], config.batch_size)
train_ulb_loader = get_data_loader(config, dataset_dict['train_ulb'], int(config.batch_size * config.uratio))
eval_loader = get_data_loader(config, dataset_dict['eval'], config.eval_batch_size)
```

이후, 다음과 같이 Training 및 Test를 진행합니다.

```python
"""5. Training"""

trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
```

```python
"""6. Test"""

trainer.evaluate(eval_loader)
```

이렇게 나온 결과들은 모두 Log에 기록되어 편하게 관리할 수 있습니다. 그렇다면, 실험 결과를 확인해 보도록 하겠습니다.

### 7-5. 실험 결과

먼저, 각 모델마다 기본적인 세팅은 다음과 같이 진행했습니다. 

- Pi-Model

```python
epoch: 5
num_train_iter: 500
num_log_iter: 50
num_eval_iter: 100
batch_size: 8
eval_batch_size: 16
num_warmup_iter: 100
# 아래의 변수들 조정
# num_labels: 
# uratio: 
ema_m: 0.0
ulb_loss_ratio: 10
unsup_warm_up: 0.4
img_size: 32
crop_ratio: 0.875
optim: AdamW
lr: 0.05
layer_decay: 0.5
momentum: 0.9
weight_decay: 0.0005
```

- Mean Teacher

```python
epoch: 5
num_train_iter: 500
num_log_iter: 50
num_eval_iter: 100
batch_size: 8
eval_batch_size: 16
num_warmup_iter: 100
# 아래의 변수들 조정
# num_labels: 
# uratio: 
ema_m: 0.999
ulb_loss_ratio: 50
unsup_warm_up: 0.4
img_size: 32
crop_ratio: 0.875
optim: AdamW
lr: 0.05
layer_decay: 0.5
momentum: 0.9
weight_decay: 0.0005
```

- VAT

```python
epoch: 5
num_train_iter: 500
num_log_iter: 50
num_eval_iter: 100
batch_size: 8
eval_batch_size: 16
num_warmup_iter: 100
# 아래의 변수들 조정
# num_labels: 
# uratio: 
ema_m: 0.0
img_size: 32
crop_ratio: 0.875
optim: AdamW
lr: 0.05
layer_decay: 0.5
momentum: 0.9
weight_decay: 0.0005
```

- UDA

```python
epoch: 5
num_train_iter: 500
num_log_iter: 50
num_eval_iter: 100
batch_size: 8
eval_batch_size: 16
num_warmup_iter: 100
# 아래의 변수들 조정
# num_labels: 
# uratio: 
ema_m: 0.0
tsa_schedule: none
T: 0.4
p_cutoff: 0.8
ulb_loss_ratio: 1.0
img_size: 32
crop_ratio: 0.875
optim: AdamW
lr: 0.05
layer_decay: 0.5
momentum: 0.9
weight_decay: 0.0005
```

- MixMatch

```python
epoch: 5
num_train_iter: 500
num_log_iter: 50
num_eval_iter: 100
batch_size: 8
eval_batch_size: 16
num_warmup_iter: 100
# 아래의 변수들 조정
# num_labels: 
# uratio: 
ema_m: 0.0
mixup_alpha: 0.5
T: 0.5
ulb_loss_ratio: 10
unsup_warm_up: 0.4
img_size: 32
crop_ratio: 0.875
optim: AdamW
lr: 0.05
layer_decay: 0.5
momentum: 0.9
weight_decay: 0.0005
```

컴퓨터의 Hardware 문제로, Batch size와 Iteration을 보통의 경우보다 낮춘 후에 실험을 진행하였습니다. 

- CIFAR-100: Label = [200, 300, 400]

![image](https://user-images.githubusercontent.com/87464956/209666750-1a78b022-c822-4403-8a45-09a0c11a7b4f.png)

CIFAR-100 data에 대해서 label data의 수를 [200, 300, 400]으로 바꿔가면서 진행한 실험 결과입니다. 처음에 설정한 가설대로, 전체적으로 Label 데이터의 수를 늘리니 성능이 좋아진 것을 확인할 수 있었습니다. 역시 Deep learning 모델에서 좋은 성능을 얻기 위해서는 Labeled data들을 기반으로 하는 Supervised learning이 잘 수행되어야 한다는 것을 알 수 있었습니다. 해당 경우, F1-Score를 기준으로 Pi-model과 VAT가 좋은 성능을 보였습니다. 그러나 Iteration을 충분히 돌리지 않아서 인지 각 모델들이 특정 모델을 제외하고는 뚜렷하게 다른 성능을 보이지 않았습니다. 

- CIFAR-100: Unlabeld ratio = [1, 2, 5]

![image](https://user-images.githubusercontent.com/87464956/209666834-91ba2ecb-fd6e-4f9c-95ee-248954373f4d.png)

CIFAR-100 data에 대해서 label data의 수는 200으로 고정한 후, Unlabeled ratio를 다음과 같이 [1, 2, 5] 순으로 올리며 진행한 실험 결과입니다. Pi-model에 대해서는 처음 설정한 가설대로 Unlabeled data의 수가 많아짐에 따라서 성능이 조금씩 하락하는 모습을 보였으나, 다른 모델들은 오히려 성능이 올라가는 모습을 보였습니다. 이에, 단순히 Unlabeled data의 비율이 높아진다고 성능이 하락하는 것이 아니라, 모델 자체적으로 Unlabeled data에 대해서 잘 Pseudo label을 부여한다면, 오히려 성능이 올라갈 수 있겠다라는 것이 맞겠다고 생각하였습니다. 이때 Uratio의 수를 증가시킴에 따라 가용해야 하는 Data의 수가 크게 증가하므로 Mixmatch(5)의 경우에는 Out of memory라는 결과를 보였습니다.

- EuroSAT: Label = [20, 30, 40]

![image](https://user-images.githubusercontent.com/87464956/209666850-cffffa2c-1dec-4332-889d-026229e280ab.png)

EuroSAT data에 대해서 label data의 수를 [20, 30, 40]으로 바꿔가면서 진행한 실험 결과입니다. 해당 Dataset은 인공위성 Data로, Class가 의미하는 것은 인공위성에서 촬영한 사진(토지)이 어떻게 이용되는 지를 의미합니다. Data 자체가 사실 쉽지 않은 Task이다 보니, 전체적으로 성능이 좋지 않음을 보였다고 생각할 수 있었습니다. 그러나 전체적으로 모델들이 Labeled data의 수가 늘어날수록 더 좋은 성능을 보였다는 것을 확인할 수 있었습니다. 

- EuroSAT: Unlabeled ratio = [1, 2, 5]

![image](https://user-images.githubusercontent.com/87464956/209666862-638f1956-761a-4b44-abfb-8aef75e54dcb.png)

CIFAR-100 data에 대해서 label data의 수는 20으로 고정한 후, Unlabeled ratio를 다음과 같이 [1, 2, 5] 순으로 올리며 진행한 실험 결과입니다. 해당 실험에서도, 특정 모델들을 제외한다면 Unlabeled ratio가 증가할수록 모델의 성능이 증가함을 알 수 있었습니다. 크게 변화가 없는 모델들도 존재하였으나, Pi-model 같은 경우에는 Unlabeled ratio가 늘어날수록 성능이 크게 향상되는 모습을 볼 수 있었습니다. 이를 통하여 Pi-model은 Unlabeled data에 대해서 Pseudo labeling을 효과적으로 진행한다는 생각이 들었습니다. 이때 Uratio의 수를 증가시킴에 따라 가용해야 하는 Data의 수가 크게 증가하므로 CIFAR-100의 경우와 같이 Mixmatch(5)의 경우에는 Out of memory라는 결과를 보였습니다.

- SVHN: Label = [40, 100, 150]

![image](https://user-images.githubusercontent.com/87464956/209666872-e3cea446-0364-4766-b5f2-334bb482fe6d.png)

SVHN Data에 대해서 label data의 수를 [40, 100, 150]으로 바꿔가면서 진행한 실험 결과입니다. SVHN Data에 대해서는 전체적으로 모델들의 성능이 매우 낮게 나왔음을 확인할 수 있었습니다. 이는 SVHN Data set이 총 73,257 개의 Training data로 이루어져 있음에도 Labeled data의 수를 너무 낮게 설정했기 때문이 아닌가라는 생각을 하였습니다. 또한, 전체적인 데이터의 수에 비하면 너무 작은 Labeled data수 이긴 하지만, Labeled data의 수를 조금씩 올렸음에도 불구하고 성능이 좋아지지 않는 모습을 확인할 수 있었습니다. 해당 부분은 Hyper parameter를 조금씩 변경 시켜가며 더 다양한 실험을 진행하였으면 분명히 더 좋은 결과를 낼 수 있지 않았을까라는 생각을 하였습니다.  

- SVHN: Unlabeled ratio = [1, 2, 5]

SVHN Data 자체가 많은 양의 Data를 가지고 있기에, Unlabeld ratio를 조금씩 늘리더라도 데이터의 수가 너무 많이 증가하기에 Batch size가 8인 경우에도 Out of memory 문제가 발생하여, 4로 내려서 다시 실험을 진행한 결과에서도 특정 결과들을 제외하면 대부분이 Out of memory 문제가 발생하여 해당 실험을 진행할 수 없었습니다.

최종적으로 **4가지의 Consistency Regularization 모델들, Pi-Model, Mean Teacher, VAT, UDA**와 **MixMatch** 를 더하여 총 5가지의 모델에 3 가지의 데이터셋, **CIFAR-100, EuroSAT, SVHN**에 대하여 2가지의 변수, **Label data의 수**와 **Unlabeled data의 비율**을 바꿔가면서 실험을 진행하였습니다. 모든 모델이 그런 것은 아니지만 예상한 대로 Label data의 수를 늘릴수록 전체적으로 모델들의 성능이 좋아짐을 알 수 있었습니다. 그러나, Unlabeled ratio를 수정하는 경우에는, 각 모델들이 일정한 결과값을 보이지는 않았습니다.  이에 Unlabeled ratio를 대폭 늘렸을 때 성능이 상승하는 모델이라면, 효과적인 Pseudo labeling을 수행할 수 있는 모델이라고 해석할 수 있었습니다.
