# Example usage

To get a trained ONNX model, please run

```
train-lenet5-mnist --output-path lenet5-mnist.onnx --num-epochs 100 --batch-size 64 --learning-rate 0.001 --random-seed 0
```

Once training finishes, you can use the script `run-inference.py` to run the model on parts of the MMIST data set:

```
python run-inference.py
```