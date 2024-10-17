<p align="center">
  <img src="images/transpad_logo.jpg" alt="Figure 1" width="100%">
  <br>
</p>

# TransPAD: Transformer for Point Anomaly Detection (author implementation)

This repository contains our implementation of **"Transformer for Point Anomaly Detection"** (CIKM2024).

## Paper Overview

In data analysis, unsupervised anomaly detection holds an important position for identifying statistical outliers that correspond to atypical behavior, erroneous readings, or interesting patterns across data.
The Transformer model [^1], known for its ability to capture dependencies within sequences, has revolutionized areas such as text and image data analysis.
However, its potential for tabular data, where sequential dependencies are not inherently present, remains underexplored.

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

We provide the network architecture of the proposed TransPAD model and share the pipeline code that enables users to train and test the network on the given dataset. 
The Transformer code used in TransPAD references the implementation provided by [lucidrains's repository](https://github.com/lucidrains/vit-pytorch). 
All experiments were conducted in an environment with `Python 3.8.12`, `PyTorch 1.12.1`, and `CUDA Toolkit 11.3.1`.

### Dataset and Hyperparameter Configuration

The parameters that users can modify, such as the paths to the training datasets and the hyperparameters, are primarily defined in `parameters.py`

- **Dataset Configuration**

As an example, we share the Lung dataset [^4] used in the experiments described in the paper (datasets/lung-1vs5.csv). 
All datasets used in the experiments must be min-max normalized per feature, with the last feature serving as a binary label distinguishing between normal and abnormal cases. 
The path to the dataset, saved as a `.csv` file, can be specified as follows:

```python
# parameters.py
dataset_root = '[PATH OF dataset.csv FILE]'
```

- **Hyperparameter Configuration**

The default hyperparameters for the model have been optimized based on the experiments conducted with the Lung dataset as described in the paper. 
These hyperparameters can also be adjusted in `parameters.py` as shown below:

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

- **Setting the Experimental Results Path**

After configuring the dataset and hyperparameters, you will need to set the path for saving the experimental results. 
This can also be done in `parameters.py` using the variables `results_path` and `exp_name`. 
`results_path` specifies the default directory where the experimental results will be saved, and `exp_name` defines the name of the current experiment.

For example, with the following configuration:
```python
# parameters.py
results_path = './results'

exp_name = 'test'
```
The best-performing model trained during the experiment will be saved as `results/test/best_auroc_model.pt`.

- **Run the Experiment**
  
If you have completely finished setting up `parameters.py`, you can start the experiment by running the following command:
```
python main.py
```

### Citation
If you utilize this code, please cite the following paper and star this repository:
```bibtex
@inproceedings{kim2024transpad,
  title={Transformer for Point Anomaly Detection},
  author={Kim, Harim and Lee, Chang Ha and Hong, Charmgil},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  year={2024},
}
```

### References

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.
[^2]: Hu Wang, Guansong Pang, Chunhua Shen, and Congbo Ma. 2019. Unsupervised representation learning by predicting random distances. arXiv preprint arXiv:1912.12186.
[^3]: Leland McInnes, John Healy, and James Melville. 1802. Umap: uniform manifold approximation and projection for dimension reduction. arxiv 2018. arXiv preprint arXiv:1802.03426.
[^4]: Z.Q. Hong and J.Y. Yang. 1992. Lung cancer. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C57596. (1992).
