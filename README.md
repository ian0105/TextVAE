# transformers를 활용한 VAE를 만들려는 시도


## 현존하는 Transformer VAE
### 1.Optimus: the first pre-trained Big VAE language model
EMNLP 2020 paper
Microsoft Research 팀 제작

문제: Interpolation이 전혀 안 됨... -> 사실상 VAE의 역할을 해낸다고 볼 수 없음


### 2. Finetuning Pretrained Transformers into Variational Autoencoders

문제: beam search대신 greedy로 출력하면 똑같은 문장만 계속 나옴 -> posterior collapse 전혀 해결하지 못함! 

