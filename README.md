# neural-network-pruning

Pruning Methods:
* global magnitude unstructured
* local magnitude unstructured
* global magnitude structured
* local magnitude structured
* global random unstructured
* local random unstructured



Experiments:

VGG-16 --> not converging, next: try VGG with Batch Normalization

ResNet34 --> converging, takes lots of time with pruning masks

LeNet5 like --> converging as expected, pruning works as expected

LeNet300-100 --> converging as expected, pruning works as expected

Use Xavier Initialization