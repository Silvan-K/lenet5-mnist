#!/usr/bin/env python3

if __name__ == "__main__":

    # Load the data from the set that was used to train the model
    from lenet5mnist.data import get_datasets
    train_set, test_set = get_datasets(batch_size=1)

    # Instantiate an onnxruntime inference session for model
    from onnxruntime import InferenceSession
    session = InferenceSession("lenet5-mnist.onnx")
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name

    # Loop over data and calculate top-1 accuracy
    n_hit, n_tot = 0.0, 0.0
    for i, (image, label) in enumerate(iter(test_set)):

        # Model returns scores for each category. Call agrmax on model output to
        # get prediction for category with top score
        pred = session.run([oname], {iname: image.numpy()})[0].argmax()
        
        n_hit += 1.0 if pred == label else 0
        n_tot += 1.0

        if (i+1)%100 == 0:
            print(f"Top-1 accuracy: {n_hit/n_tot}")
