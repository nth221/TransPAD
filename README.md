# TransPAD: Transformer for Point Anomaly Detection

This repository is for CIKM 2024 paper **"Transformer for Point Anomaly Detection"**.

## Paper Overview

In data analysis, unsupervised anomaly detection holds an important position for identifying statistical outliers that signify atypical behavior, erroneous readings, or interesting patterns across data.
The Transformer model [^1], known for its ability to capture dependencies within sequences, has revolutionized areas such as text and image data analysis.
However, its potential for tabular data, where sequence dependencies are not inherently present, remains underexplored.

In this paper, we introduce a novel Transformer-based AutoEncoder framework, _TransPAD_ (Transformer for Point Anomaly Detection).
Our method captures interdependencies across entire datasets, addressing the challenges posed with non-sequential, tabular data.
It incorporates unique random and criteria sampling strategies for effective training and anomaly identification, and avoids the common pitfall of trivial generalization that affects many conventional methods.
By leveraging an attention weight-based anomaly scoring system, _TransPAD_ offers a more precise approach to detect anomalies.

## Supplementary Experimental Results

<p align="center">
  <img src="images/MNIST_synt.png" alt="Figure 1" width="50%">
  <br>
  Figure 1
</p>

As shown in Figure 1-(a), The paper demonstrates that anomaly localization can be achieved by utilizing the Transformer’s attention weights as anomaly scores. Additionally, it presents in the preliminaries that frame-level anomaly detection, such as anomaly detection in tabular datasets, is possible using a novel approach called random/criteria sampler (Figure 1-(b)).

In the experiments, TransPAD was compared against existing anomaly detection methods across 10 benchmark tabular datasets. The results showed that TransPAD achieved up to a 24% improvement in AUROC (Area Under the Receiver Operating Characteristic Curve) compared to RDP (Random Distance Prediction) [^2], which was the best-performing method among the existing unsupervised point anomaly detection methods.

<p align="center">
  <img src="images/umap_visualizations.jpg" alt="Figure 2" width="80%">
  <br>
  Figure 2
</p>

Moreover, to understand the prediction patterns and mechanisms of the model in the embedding space, UMAP (Uniform Manifold Approximation and Projection) [^3] was used to visualize the data embeddings at each encoder layer of TransPAD in a two-dimensional space. Additional visualization results are shared in this repository (Figure 2).

## Experimental Setup

우리는 제안하는 TransPAD의 네트워크 구조와 해당 네트워크를 주어진 데이터셋에 대해 학습과 검출을 해볼 수 있는 파이프라인 코드를 공유한다. 
우리는 `python 3.8.12`, `pytorch 1.12.1`, `cudatoolkit 11.3.1` 버전 환경에서 실험을 진행하였다.

### 데이터셋 및 하이퍼파라미터 설정

실험에 필요한 데이터셋 경로와 모델의 학습 하이퍼파라미터 등 사용자가 조작할 수 있는 기본적으로 `parameters.py`에 정의되어 있다.

- 데이터셋 설정
  
우리는 예를 들기 위해 논문의 실험에 사용된 Lung dataset [^4]을 기본적으로 공유한다 (datasets/lung-1vs5.csv).
실험에 사용되는 데이터셋은 모두 각 feature를 기준으로 min-max normalization 되어야 하며, 마지막 feature는 normal과 anomaly를 구별하는 이진 레이블로 구성되어야 한다.
이후 `.csv` 확장자로 저장된 데이터셋의 경로를 아래와 같이 지정한다.

```python
# parameters.py
dataset_root = '[PATH OF dataset.csv FILE]'
```

- 하이퍼파라미터 설정

모델의 default hyperparameters는 Lung dataset에 대한 논문의 실험에서 최적화된 하이퍼파라미터를 적용하였다.
하이퍼파라미터는 마찬가지로 `parameters.py`에서 아래와 같이 수정 가능하다.

```python
# parameters.py
hp = {
    'batch_size' : [BATCH SIZE],
    'lr' : [LEARNING RATE],
    'sequence_len' : [SEQUENCE LENGTH],
    'heads' : [NUMBER OF HEADS],
    'dim' : [ENCODER'S INPUT DIMENSION],
    'num_layers' : [NUMBER OF LAYERS],
    'layer_conf' : [LAYER CONFIGURATION: {'same', 'smaller', 'hybrid'} OPTIONS ARE AVAILABLE] 
}
```

- 실험 결과 경로 설정

데이터셋과 하이퍼파라미터에 대한 설정이 끝난 이후에는 실험 결과에 대한 경로 설정이 필요하다.
이 역시 `parameters.py`에서 `results_path`와 `exp_name`이라는 변수를 통해 설정 가능하다.
`results_path`는 실험의 결과들이 저장되는 기본 경로를 의미하고, `exp_name`은 이번 실행할 실험의 이름을 의미한다.

예를 들어 아래와 같이 설정한 경우,
```python
# parameters.py
results_path = './results'

exp_name = 'test'
```
실험 동안 학습된 최고성능 모델은 `results/test/best_auroc_model.pt` 로 저장된다. 




### References

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.
[^2]: Hu Wang, Guansong Pang, Chunhua Shen, and Congbo Ma. 2019. Unsupervised representation learning by predicting random distances. arXiv preprint arXiv:1912.12186.
[^3]: Leland McInnes, John Healy, and James Melville. 1802. Umap: uniform manifold approximation and projection for dimension reduction. arxiv 2018. arXiv preprint arXiv:1802.03426.
[^4]: Z.Q. Hong and J.Y. Yang. 1992. Lung cancer. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C57596. (1992).
