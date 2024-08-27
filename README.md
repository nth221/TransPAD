# TransPAD: Transformer for Point Anomaly Detection

This repository is for CIKM 2024 paper **"Transformer for Point Anomaly Detection"**.

## Introduction

In data analysis, unsupervised anomaly detection holds an important position for identifying statistical outliers that signify atypical behavior, erroneous readings, or interesting patterns across data.
The Transformer model [^1], known for its ability to capture dependencies within sequences, has revolutionized areas such as text and image data analysis.
However, its potential for tabular data, where sequence dependencies are not inherently present, remains underexplored.

In this paper, we introduce a novel Transformer-based AutoEncoder framework, _TransPAD_ (Transformer for Point Anomaly Detection).
Our method captures interdependencies across entire datasets, addressing the challenges posed with non-sequential, tabular data.
It incorporates unique random and criteria sampling strategies for effective training and anomaly identification, and avoids the common pitfall of trivial generalization that affects many conventional methods.
By leveraging an attention weight-based anomaly scoring system, _TransPAD_ offers a more precise approach to detect anomalies.



## References

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30
