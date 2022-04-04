# Deep Neural Network Pruning and Adversarial Robustness

This Repository contains the code for the paper ['Pruning in the face of adversaries'](https://arxiv.org/abs/2108.08560) examining the impact of neural network pruning on the adversarial robustness of deep neural networks.

The pipelines directory contains five jupyter notebooks with which the experiments can be replicated. We implement five DNN architectures (MLP-300-100, LeNet-5-like, ResNet8, VGG11, and ResNet18), six pruning methods (structured/unstructured, global/local, random/magnitude-based) and evaluate the models' robustness against three attacks in different distance metrics ($L_0$-Carlini&Wagner, $L_2$-Brendel&Betghe, and $L_\infty$-PGD) with several $\epsilon$-values. 

### Environment
The most important packages and versions are:
- Python 3.8.3
- Tensorflow 2.2
- Tensorflow-datasets 3.1.0
- Foolbox 3.0.4

### Results
You can find a write-up of the results on [arxiv](https://arxiv.org/abs/2108.08560)
