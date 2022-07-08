# Overview

An implementation of [LeNet-5](https://en.wikipedia.org/wiki/LeNet) in pytorch,
readily packaged to be trained on the MNIST data set.

# Installation

To install, please run

```
pip install .
```

# Usage

To get a trained ONNX model, please run e.g.

```
train-lenet5-mnist --output-path lenet5-mnist.onnx --num-epochs 100 --batch-size 64 --learning-rate 0.001 --random-seed 0
```