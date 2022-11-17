# 목차

1. 이상치 탐지란?
2. Autoencoder
    - 2-1. Tutorial
    - 2-2. Experiments




## 1. 이상치 탐지(Anomaly Detection)란?

먼저, **Machine Learning**은 크게 **Supervised Learning**과 **Unsupervised Learning**으로 나눌 수 있습니다. 

![image](https://user-images.githubusercontent.com/87464956/202420895-efa1fb84-2845-43e6-9810-2b0346ba327f.png)

이러한 특성을 가지므로, **하고자 하는 Task의 목적과 Data에 가장 잘 맞는 방법론**을 선택하여 사용하시면 됩니다.  

![image](https://user-images.githubusercontent.com/87464956/202421328-d91dc37e-21a3-4d2b-8503-081edbbbf945.png)

이때, 저희가 이번에 알아보고자 하는 것은 바로 **이상치 탐지(Anomaly Detecion)** 입니다. 

**이상치(Anomaly)데이터**는 **정상(Normal)** 적인 분포에서는 **매우 낮은 확률로 나타나는 Data** 입니다. 이러한 이상치를 찾는 이유는 대부분의 데이터가 한 범주에 속하며, 극소수의 데이터만 다른 범주에 속하는 문제를 해결하기 위함입니다. 예를 들면, 다음과 같은 사례가 존재합니다. 

- 제조업 공정에서의 불량 탐지
- 신용카드 사기 거래 탐지
- 통신망의 불법적인 이용

위의 왼쪽 예시를 보시면, $N_1$ , $N_2$ 데이터의 분포와 $O_1$, $O_2$, $O_3$ 데이터의 분포가 확연히 다르므로,  $O_1$, $O_2$, $O_3$ 데이터들을 이상치라고 판단할 수 있습니다. 또한, 오른쪽의 시계열 데이터같은 경우에는 정상적인 패턴에서 벗어나는 부분을 이상치라고 판단합니다.

이때 잘 구분하셔야 하는 것이, **이상치는 노이즈 데이터와는 다르다는 점**입니다. 노이즈는 측정 과정에서의 무작위성(Randomness)에 기반하는 데이터를 말하며, 실제 현업에서도 이러한 무작위성이 존재하므로 노이즈 데이터를 꼭 제거해야하는 나쁜 데이터라고는 말할 수 없습니다. 이상치 데이터는 정상적인 데이터를 생성하는 매커니즘을 위반하여 생성되는 데이터 이므로, 이들을 구분할 수 있어야 합니다.

![image](https://user-images.githubusercontent.com/87464956/202421677-e49fe5e9-5e50-4f84-afab-4ce8a446b397.png)

**이상치를 분류하는 문제는 Classification task와 다음과 같은 차이점**을 가집니다. 이처럼 Classification을 한 후 새로운 데이터 A, B가 들어온다면, A, B는 파란색 동그라미로 Classification 됩니다. 그러나 오른쪽의 이상치 탐지 문제에서는, A, B가 들어와도 정상으로 분류되지 않습니다.

그렇다면, 실제 Task에서, Classification와 Anomaly detection 중 어떤 방법을 사용해야 할까요 ? 이는 다음과 같은 Proccess를 따라가게 됩니다.  

![image](https://user-images.githubusercontent.com/87464956/202421702-a81df12d-bdfd-42b8-9d7d-9f7cf9610da4.png)

먼저 **Class에 불균형이 존재하는지 여부**를 판단합니다. 이때, Class의 불균형이 2:8이나, 1:9 정도의 경우에는 그렇게 심한 경우는 아니고, 1:99 정도는 되어야지 불균형이 존재한다고 판단합니다. 이후에는 **소수 Class에 대해 절대적인 관측치가 어느 정도 존재하는 가에 대해서 판별**하여, Classification와 Anomaly detection 을 결정하게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/202421731-8bedff29-82b1-40be-a979-8c0b69b310b8.png)

이상치 탐지에서는, **주어진 데이터로부터 정상 범주의 개념을 확장하는 일반화(Generalization)** 와, **주어진 데이터로부터 정상 범주의 개념을 좁혀가는 특수화(Specialization)** 에서의 **Trade-off** 를 잘 고려해야 합니다. 일반화에 너무 치중한다면 이상치 데이터 판별이 어렵게 되고, 특수화에 치중한다면 빈번한 False alarm으로 과적합에 빠질 수 있습니다.

![image](https://user-images.githubusercontent.com/87464956/202421758-e0832562-9316-4571-85f3-8f57f452e9a7.png)

이때, 이상치 탐지는 다음과 같이 **다양한 방식**으로 진행할 수 있습니다. 각 방법론을 사용할 때, 다음과 같은 방식으로 학습을 진행합니다.



## 2. Autoencoder

해당 Tutorial에서는 **Autoencoder**를 통하여 이상치 탐지를 진행할 예정입니다.

![image](https://user-images.githubusercontent.com/87464956/202421888-06e0d5b6-0209-4ff2-813c-3ba018b1f9ed.png)

**Autoencoder**는 다음과 같이 입력과 출력이 동일한 신경망 구조를 가지고 있습니다. 

먼저 입력 데이터를 **Encoder**에 넣습니다. 이후 **BottleNeck**에서 **해당 데이터의 특징을 최대한 보존하면서, 데이터의 차원은 축소된 Representation**을 출력한 후, 해당 Representation을 통하여 **Decoder**에서는 원래의 입력 이미지를 다시 재구성합니다. 

이러한 목적을 이루기 위해서 Bottle Neck 부분은 반드시 입력 레이어보다는 더 적은 노드로 구성해야 하며, Representation은 **해당 데이터의 중요한 정보를 가지지만, 차원은 축소되어 있는 정보인 Latent vector**로 표현됩니다. 

그런데, 이때 의문점이 하나 있습니다. 입력과 출력이 동일하다면, 이러한 신경망 구조를 왜 학습하는 것일까요?

![image](https://user-images.githubusercontent.com/87464956/202421912-8a92054a-bc4b-4bc2-b7d1-2a1498e66b42.png)

**이상치 탐지를 Autoencoder에 적용** 시키려면 다음과 같은 방법을 생각할 수 있습니다. 

먼저 **충분한 양의 정상 데이터로만  Autoencdoer를 학습**시켜서, BottleNeck이 **입력 데이터에 대한 Representation을 잘 학습**하도록 합니다. 이후 새로운 데이터가 들어오면, 다음과 같은 두 가지를 예상할 수 있습니다.

- **정상 데이터**가 들어온 경우, 정상 데이터는 **자기 자신의 데이터를 잘 복원**.
- **비정상 데이터**가 들어온 경우, 비정상 데이터에 대해서는 학습 기회가 적었으므로 정상 데이터가 들어온 경우보다 **자기 자신을 더 복원하지 못하도록 학습**.

최종적으로, **입력 정보와 Autoencoder를 통하여 복원된 출력 정보 간의 차이**를 이용하여 Anomaly score가 산출되고, Thresholding을 통하여 Anomaly을 결정합니다.

![image](https://user-images.githubusercontent.com/87464956/202421935-55d1dda8-3c56-4d7f-ae07-d0f7830249fe.png)

그러나, 이러한 Autoencoder를 통한 Anomaly detection에는 약간의 **문제점**이 존재합니다. 이는 Autoencoder 뿐만 아닌 Reconstruction 기반의 Anomaly detection 방법들이 가지는 공통적인 문제인데, 다음의 예시와 같이 **정상 데이터만 복원을 잘 해야 하는데, 이상 데이터도 복원을 잘 한다는 것입니다.** 이러한 문제는 학습 과정에서 일부러 Noise를 첨가하여 해결할 수 있습니다.

### 2-1. Tutorial

해당 Tutorial에서는 Autoencdoer를 통하여 이상치 탐지를 진행할 예정입니다. 이때, 해당 실험에서는 다음의 두 가지 데이터를 이용하여 실험을 진행하고자 합니다. 

- **난수를 통해 만든 Custom data**
    - 정상 데이터: 기대값은 0이고, 표준편차가 1인 가우시안 표준 정규 분포를 따른 난수들을 생성한 후, 약간의 변형을 가한 값들
    - 비정상 데이터: -3.5에서 3.5 사이의 uniform distribution에서 랜덤 추출한 값
- **MNIST 데이터**
    - 정상 데이터: 기존 MINIST 데이터
    - 비정상 데이터: MNIST 크기(28*28) 만큼의 픽셀에 노이즈를 생성한 값과 기존 MNIST 데이터들의 배경에 노이즈를 생성한 데이터들

또한, 제가 이번 실험은 다음과 같은 **Hyper parameter**의 변화를 통하여 모델의 학습이 어떻게 바뀌는지 확인하고자 합니다.

- **이상치 데이터의 개수**
- **learning rate**
- **Loss function**
- **Autoencdoer 내의 Layer 수**

먼저, 해당 실험을 위해 필요한 데이터들을 만들어 줍니다.

```python
"""데이터 1. Custom 데이터"""

# Random 수치 설정
random_state = np.random.RandomState(42)

# 전체 데이터 수와 이상치 비율 설정
num_made_data = 10000
test_made_rate = 0.2
anomal_made_rate = 0.02

# X_made_train 설정 
X_made_train = 0.2 * random_state.randn(num_made_data, 2)
X_made_train = np.r_[X_made_train+2, X_made_train-2]
X_made_train = pd.DataFrame(X_made_train, columns = ['x1', 'x2'])

# X_made_test 설정
X_made_test = 0.2 * random_state.randn(int(num_made_data * test_made_rate), 2)
X_made_test = np.r_[X_made_test+2, X_made_test-2]
X_made_test = pd.DataFrame(X_made_test, columns = ['x1', 'x2'])

# 이상치 데이터 생성
abnormal_made_data = random_state.uniform(low=-3.5, high=3.5, size=(int(num_made_data * anomal_made_rate), 2))
abnormal_made_data = pd.DataFrame(abnormal_made_data, columns = ['x1', 'x2'])

# Train과 이상치 데이터 시각화
plt.scatter(X_made_train.x1, X_made_train.x2, c='white', s=10*4, edgecolor='k', label='Normal')
plt.scatter(abnormal_made_data.x1, abnormal_made_data.x2, c='red', s=10*4, edgecolor='k', label='Abnormal')

plt.legend(loc='lower right')
plt.show()
```

먼저 난수를 통한 Custom data를 만들어 줍니다. 이때, 전체 데이터의 수는 10,000개로 하고, Train data와 Test data의 비율을 8:2로 설정합니다. 이후 이상치 데이터는 다음과 같이 전체 데이터의 2%, 즉 200개의 이상치 데이터를 만듭니다.

![image](https://user-images.githubusercontent.com/87464956/202422020-fcbee157-34b2-4ff4-9133-7df0fb3aefb8.png)

```python
# 이상치, 정상데이터에 라벨 추가한 후, Test_data에 이상치 섞어주기

X_made_train['label'] = 0
X_made_test['label'] = 0
abnormal_made_data['label'] = 1

X_made_test = pd.concat([X_made_test, abnormal_made_data], axis = 0)
X_made_test = X_made_test.sample(frac=1)
```

이후, 이렇게 만들어진 이상치 데이터와 정상 데이터에 비교를 위한 label을 추가해준 후, 평가를 위하여 Test data는 이상치 데이터와 합친 후에 랜덤하게 배치 시키도록 합니다.

```python
"""데이터 2. MNIST 데이터에 임의로 이상치 만들어서 넣어주기"""
mnist = fetch_openml('mnist_784')

# 비율 설정
rate_mnist = 0.1
test_mnist_rate = 0.2
anomal_mnisit_rate = 0.01

# Train, Test 설정
X_mnist= mnist['data'][:int(len(mnist['data'])*rate_mnist)]

# Train, Test 비율 설정
X_mnist_train, X_mnist_test = X_mnist[:int(len(X_mnist)*(1-test_mnist_rate))], X_mnist[int(len(X_mnist)*(1-test_mnist_rate)):]
mnist_col = X_mnist_train.columns

# 이상치 데이터 비율 설정
abnormal_mnist = mnist['data'][len(X_mnist):len(X_mnist)+int(len(X_mnist)*(anomal_mnisit_rate))]
```

다음으로는, 또 다른 데이터인 MNIST 데이터를 생성합니다. scikit-learn에서 제공하는 MNIST 데이터는 70000장이 있으므로, 이 중 7000장 만 사용하기 위하여 비율을 설정해줍니다. 이후 Test data, 이상치 데이터 역시 다음처럼 비율을 설정하여 불러옵니다.

```python
# 기존 Mnist와는 아예 다른 이미지 생성

def make_fake_img(n_fake_img):
    img_size = 28
    n_fake_img = n_fake_img
    fake_img  = []
    for i in range(n_fake_img):
        fake_img.append(np.random.randn(img_size * img_size).reshape(1, img_size, img_size) )
 
    fake_img = torch.FloatTensor(fake_img)
    fake_img = fake_img.view(n_fake_img, img_size * img_size)
 
    return fake_img

random_mnist_anormal_or = make_fake_img(len(abnormal_mnist))
random_mnist_anormal = pd.DataFrame(random_mnist_anormal_or)
random_mnist_anormal.columns = mnist_col

# Mnist 데이터 set에서의 anomal 생성

for i in range(len(abnormal_mnist)):
    row = abnormal_mnist.iloc[i]
    for i in range(len(row)-1):
        row[i+1] = min(255, row[i+1]+random.randint(100,200))
```

이후, 다음과 같이 MNIST와 size는 동일하지만 모든 픽셀이 노이즈로 이루어진 이상치 데이터와, 기존 MNIST 데이터에 Perturbation 가하여 이상치 데이터를 생성한 후, 이들을 결합하여 사용합니다.

```python
# 정상데이터 시각화

digit = X_mnist_train.iloc[3].to_numpy()
digit_image = digit.reshape(28, 28)

plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.show()

# 랜덤하게 생성한 데이터 시각화

digit = random_mnist_anormal.iloc[0].to_numpy()
digit_image = digit.reshape(28, 28)

plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.show()

# Mnist 데이터에 변형 준 데이터 시각화

digit = abnormal_mnist.iloc[3].to_numpy()
digit_image = digit.reshape(28, 28)

plt.imshow(digit_image, cmap="binary")
plt.axis("off")
plt.show()
```

해당 이미지들을 모두 시각화 해보면 차례대로 다음과 같은 결과가 나옵니다.

![image](https://user-images.githubusercontent.com/87464956/202422053-a07e29ed-3166-4f1e-99c6-94b43217e56f.png)

```python
# 정상, 이상치 데이터 라벨링. Anomal 데이터 합쳐준 후, Train에는 정상 데이터만, Test에는 정상 + 이상 데이터 함께 구성

X_mnist_train['label'] = 0
Anomal_data_mnist = pd.concat([abnormal_mnist, random_mnist_anormal], axis = 0)
Anomal_data_mnist['label'] = 1
X_mnist_test['label'] = 0
X_mnist_test = pd.concat([X_mnist_test, Anomal_data_mnist], axis = 0, ignore_index=True)
X_mnist_test = X_mnist_test.sample(frac=1)
```

이후, 이렇게 만들어진 이상치 데이터와 정상 데이터에 비교를 위한 label을 추가해준 후, 평가를 위하여 Test data는 이상치 데이터와 합친 후에 랜덤하게 배치 시키도록 합니다.

```python
# 직접 만든 데이터이므로, Cumstom data 설정 해주는 함수.

class Mnist_Loader(Dataset):
    def __init__(self):
        super(Mnist_Loader, self).__init__()
        self.dataset = ''
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row)/255).float()
        return data
    
class custom_Loader(Dataset):
    def __init__(self):
        super(custom_Loader, self).__init__()
        self.dataset = ''

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row)).float()
        return data

# MNIST 데이터셋 구성 
 
class Mnist_Train_Loader(Mnist_Loader):
    def __init__(self):
        super(Mnist_Train_Loader, self).__init__()
        self.dataset = X_mnist_train
        
class Mnist_Test_Loader(Mnist_Loader):
    def __init__(self):
        super(Mnist_Test_Loader, self).__init__()
        self.dataset = X_mnist_test
        
trainset_mnist = Mnist_Train_Loader()
testset_mnist = Mnist_Test_Loader()

train_mnist = DataLoader(
            trainset_mnist,
            batch_size=32,
            shuffle=True
        )

test_mnist = DataLoader(
            testset_mnist,
            batch_size=32,
            shuffle=False
        )

# Custom 데이터셋 구성

class made_Train_Loader(custom_Loader):
    def __init__(self):
        super(made_Train_Loader, self).__init__()
        self.dataset = X_made_train
        
class made_Test_Loader(custom_Loader):
    def __init__(self):
        super(made_Test_Loader, self).__init__()
        self.dataset = X_made_test
        
trainset_made = made_Train_Loader()
testset_made = made_Test_Loader()

train_made = DataLoader(
            trainset_made,
            batch_size=32,
            shuffle=True
        )

test_made = DataLoader(
            testset_made,
            batch_size=32,
            shuffle=False
        )
```

이렇게 만들어진 데이터들은 모두 직접 만든 데이터이므로, Pytorch 프레임워크를 통해 구현한 Autoencdoer에 넣기 위해서는 데이터 셋으로 구성한 후, 이를 loader한 후 모델에 집어 넣을 수 있게 됩니다.

```python
"""Original Autoencoder: Mnist"""

class Mnist_encoder(nn.Module):
    def __init__(self):
        super(Mnist_encoder, self).__init__()
        self.Mnist_encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(), 
            nn.Linear(32, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
 
    def forward(self, x):
        x = self.Mnist_encoder(x)
        return x
 
 
class Mnist_decoder(nn.Module):
    def __init__(self):
        super(Mnist_decoder, self).__init__()
        self.Mnist_decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 28*28),
            nn.BatchNorm1d(28*28),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.Mnist_decoder(x)
        return x

class Mnist_decoder(nn.Module):
    def __init__(self):
        super(Mnist_decoder, self).__init__()
        self.Mnist_decoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(), 
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Linear(512, 28*28),
            nn.BatchNorm1d(28*28),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        x = self.Mnist_decoder(x)
        return x
```

이후, 각 데이터의 특성에 맞는 Autoencoder를 구성합니다. 

- MNIST:
    - Encoder: MNIST 데이터는 28*28 크기를 가지므로,  28*28의 input size에서 2까지 축소하여 28*28 size의 데이터 정보를 압축한 Latent vector 생성. 이때, 각 단계마다 Batch Normalization과 ReLU Activation function 적용
    - Decoder:  Encoder에서 만들어진 Representation인 Latent vector를 입력으로 받아서 다시 28*28 size의 데이터로 만들어 준다.
- Custom Data:
    - Encoder: Custom 데이터는 2개의 Feature만으로 이루어진 데이터이기에 2개를 1개로 줄이면서 데이터 정보를 압축하고, Representation을 가지는 Latent vector 생성. 이때, Batch Normalization과 ReLU Activation function 적용
    - Decoder: Encoder에서 만들어진 Representation인 Latent vector를 입력으로 받아서 다시 2size 의 데이터로 만들어 준다.

```python
# GPU 할당
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 인코더 디코더 설정
Mnist_encoder = Mnist_encoder().to(device)
Mnist_decoder = Mnist_decoder().to(device)

made_encoder = made_encoder().to(device)
made_decoder = made_decoder().to(device)

learning_rate = 0.001
num_epochs = 100
criterion = nn.MSELoss()
w_d = 0.0001
metrics = defaultdict(list)

Mnist_encoder_optimizer = torch.optim.Adam(Mnist_encoder.parameters(), lr=learning_rate, weight_decay=w_d)
Mnist_decoder_optimizer = torch.optim.Adam(Mnist_decoder.parameters(), lr=learning_rate, weight_decay=w_d)

made_encoder_optimizer = torch.optim.Adam(made_encoder.parameters(), lr=learning_rate, weight_decay=w_d)
made_decoder_optimizer = torch.optim.Adam(made_decoder.parameters(), lr=learning_rate, weight_decay=w_d)
```

이후  Training을 하기 위하여 다음과 같이 준비를 해 줍니다.

```python
# MNIST 훈련

Mnist_encoder.train()
Mnist_decoder.train()

start = time.time()
print('-----------------Original Autoencoder: MNIST----------------')
for epoch in range(num_epochs):
    ep_start = time.time()
    # running_loss = 0.0
    
    for bx, (data) in enumerate(train_mnist):
        img = data
        img = img.view(img.size(0), -1)
        img = Variable(img)

        latent_z = Mnist_encoder(img.to(device))
        output = Mnist_decoder(latent_z.to(device))

        loss = criterion(output, img.to(device))
 
        Mnist_encoder_optimizer.zero_grad()
        Mnist_decoder_optimizer.zero_grad()
        loss.backward()
        Mnist_encoder_optimizer.step()
        Mnist_decoder_optimizer.step()
        # running_loss += loss.item()
        
    # epoch_loss = running_loss/len(trainset_mnist)
    metrics['train_loss'].append(loss.data)
    ep_end = time.time()
    print('-------------------------------------------------------------')
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
end = time.time()
print('-------------------------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))

# Custom data 훈련

made_encoder.train()
made_decoder.train()

start = time.time()
print('-----------------Original Autoencoder: Custom----------------')
for epoch in range(num_epochs):
    ep_start = time.time()
    # running_loss = 0.0
    
    for bx, (data) in enumerate(train_made):
        img = data # label은 가져오지 않는다.
        img = img.view(img.size(0), -1)
        img = Variable(img)

        latent_z = made_encoder(img.to(device))
        output = made_decoder(latent_z.to(device))

        loss = criterion(output, img.to(device))
 
        made_encoder_optimizer.zero_grad()
        made_decoder_optimizer.zero_grad()
        loss.backward()
        made_encoder_optimizer.step()
        made_decoder_optimizer.step()
        # running_loss += loss.item()
        
    # epoch_loss = running_loss/len(trainset_mnist)
    metrics['train_loss'].append(loss.data)
    ep_end = time.time()
    print('-------------------------------------------------------------')
    print('epoch [{}/{}], loss:{:.4f}' .format(epoch + 1, num_epochs, float(loss.data) ))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
end = time.time()
print('-------------------------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))
```

Training 과정은 다음과 같이 정상 데이터만을 통하여 이루어집니다. 

```
# Loss graph plot

A = metrics['train_loss']
metrics_list = []
for i in range(len(A)):
    metrics_list.append(A[i].to('cpu'))
    
_, axs = plt.subplots(1 , 1 , figsize=(8,6) )

axs.plot(metrics_list , 'r', linewidth = 2)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Training Loss', fontsize=20)
```

![image](https://user-images.githubusercontent.com/87464956/202422122-fe8e46f4-1d18-43bc-9a19-81ca4931fc98.png)

똑같은 세팅 값을 바탕으로 훈련한 각 Data별 모델의 Loss 변화는 다음과 같습니다. 

이때, Custom dataset같은 경우는 Loss가 들쭉날쭉한 것을 확인할 수 있는데, learning rate가 크게 잡혀서 수렴을 하지 못하는 것인가라는 생각이 들어 이후 실험 페이지에서 바꿔가면서 실험해볼 계획입니다.

```python
# Mnist data 평가

Mnist_encoder.eval()
Mnist_decoder.eval()

loss_dist = []
pred_dist = []
correct=0

for i in range(len(X_mnist_test)):
    data = torch.from_numpy(np.array(X_mnist_test.iloc[i][:-1])).float().unsqueeze(0)
    latent_z = Mnist_encoder(data.to(device))
    output = Mnist_decoder(latent_z.to(device))
    loss = criterion(data.to(device), output)
    loss_dist.append(loss.item())

# custom data 평가

made_encoder.eval()
made_decoder.eval()

loss_dist = []
pred_dist = []
correct=0
# anom = X_mnist_test
# for bx, data in enumerate(test_made):
for i in range(len(X_made_test)):
    data = torch.from_numpy(np.array(X_made_test.iloc[i][0:2])).float().unsqueeze(0)
    latent_z = made_encoder(data.to(device))
    output = made_decoder(latent_z.to(device))
    loss = criterion(data.to(device), output)
    pred_dist.append(data.cpu().numpy().flatten())
    loss_dist.append(loss.item())
```

이후, 정상 데이터와 이상치 데이터가 포함되어 있는 Test data를 통하여 해당 모델을 평가할 수 있습니다.

```python
# Custom data 결과

lower_threshold1 = 0.5
upper_threshold1 = 2.2
lower_threshold2 = 3.0
upper_threshold2 = 5.9

plt.figure(figsize=(12,6))
plt.title('Loss Distribution')
sns.distplot(loss_dist,bins=100,kde=True, color='blue')
plt.axvline(upper_threshold1, 0.0, 1, color='r')
plt.axvline(lower_threshold1, 0.0, 1, color='b')
plt.axvline(upper_threshold2, 0.0, 1, color='r')
plt.axvline(lower_threshold2, 0.0, 1, color='b')
```

![image](https://user-images.githubusercontent.com/87464956/202422226-9ddfed3c-4869-4122-9585-37d39ad10901.png)

Threshold를 구현하는 코드는 작성하지 못해서, 해당 데이터의 분포를 보고 직접 설정하였습니다.

```python
lower_threshold1 = 0.5
upper_threshold1 = 2.2
lower_threshold2 = 3.0
upper_threshold2 = 5.9

loss_sc = []
for i in loss_dist:
    loss_sc.append((i,i))
plt.scatter(*zip(*loss_sc))
plt.axvline(upper_threshold1, 0.0, 1, color='r')
plt.axvline(lower_threshold1, 0.0, 1, color='b')
plt.axvline(upper_threshold2, 0.0, 1, color='r')
plt.axvline(lower_threshold2, 0.0, 1, color='b')
```

![image](https://user-images.githubusercontent.com/87464956/202422256-7e96f555-b62e-4023-9274-c1f22c9c5650.png)

```python

correct1 = sum(l < int(upper_threshold1) and l > int(lower_threshold1) for l in loss_dist)
correct2 = sum(l < int(upper_threshold2) and l > int(lower_threshold2) for l in loss_dist)

print(f'Correct normal predictions: {correct1 + correct2}/{len(X_made_test)}')

up_abnormal = sum(l >= int(upper_threshold2) for l in loss_dist)
middle_abnormal = sum(l >= int(upper_threshold1) and l <= int(lower_threshold2) for l in loss_dist)
down_abnormal = sum(l <= int(lower_threshold1) for l in loss_dist)

print(f'Correct anomaly predictions: {up_abnormal+middle_abnormal+down_abnormal}/{len(X_ma
de_test)}')
```

![image](https://user-images.githubusercontent.com/87464956/202422276-8e695d2f-c3a2-4826-b0c1-a50a880c4010.png)

다음의 코드를 통하여,  모델이 총 데이터 중에서 정상 데이터와 이상치 데이터를 어떻게 구분하였는지 알 수 있습니다.

```python
def get_classifier_eval(tn=tn, fp=fp, fn=fn, tp=tp):
    accuracy = (tp+tn)/(tp+fn+fp+tn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2*precision*recall)/(precision+recall)
    print(f'Accuracy:{accuracy:.4f}, Precision :{precision:.4f}, Recall:{recall:.4f}, F1-Score:{f1:.4f}')

tp, fp, tn, fn = 0, 0, 0, 0

for i in range(len(loss_dist)):
    a = X_made_test.iloc[i]
    a['loss'] = loss_dist[i]
    if a['loss'] >= upper_threshold2 or upper_threshold1 <= a['loss'] <= lower_threshold2 or a['loss'] <= lower_threshold1:
        if float(a['label']) == 1.0:
            tp += 1
        else:
            fp += 1
    else:
        if float(a['label']) == 1.0:
            fn += 1
        else:
            tn += 1
            
print('[TP] {}\t[FP] {}\n[TN] {}[FN] {}'.format(tp, fp, tn, fn))

conf = [[tp,fn],[fp,tn]]
plt.figure()
sns.heatmap(conf, annot=True, annot_kws={"size": 20}, fmt='d', cmap='GnBu')
get_classifier_eval(tn, fp, fn, tp)
print('[TP] {}\t[FP] {}\t[TN] {} [FN] {}'.format(tp, fp, tn, fn))
```

해당 코드를 통하여 TP, FP, TN, FN을 시각화 하였고, 해당 모델의 전체적인 Accuracy, Precision, Recall, F1-Score는 다음과 같이 나오게 됩니다.

![image](https://user-images.githubusercontent.com/87464956/202422294-e5bb3020-f876-4d05-8483-94297e81a23d.png)


### 2-2. Experiments

각 **Hyper parameter** 변화에 따른 실험 결과입니다. 초기 설정에서 epoch만 20으로 한 후 나머지 설정들은 그대로 유지한 채, 다음의 항목들에만 변화를 주었습니다. 

#### 이상치 비율에 따른 변화



#### Learning rate에 따른 변화



#### Loss function에 따른 변화



#### Layer에 따른 변화



