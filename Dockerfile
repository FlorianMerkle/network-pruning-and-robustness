FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

RUN pip install matplotlib foolbox tensorflow_datasets numpy pandas tqdm
