# Example usage

To get a trained ONNX model, please run

```
train-lenet5-mnist --output-path lenet5-mnist.onnx --num-epochs 100 --batch-size 64 --learning-rate 0.001 --random-seed 0
```

Once training finishes, you can use the script [run-inference.py](https://github.com/Silvan-K/lenet5-mnist/blob/dd7f1f718238875d73f61a02444a4c50d8aa0ae2/example/run-inference.py) to run the model on parts of the MMIST data set:

```
python run-inference.py
```
