import torch

def train_model(model, dataset, loss_fct, optimizer, num_epochs):

    import json

    # Determine device we are running on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # To monitor loss
    loss_history = []
    
    # Traning loop
    num_batches = len(dataset)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataset):

            # Run model on current batch and evaluate loss
            images  = images.to(device)
            labels  = labels.to(device)
            outputs = model(images)
            loss    = loss_fct(outputs, labels)
        	
            # Evaluate gradient and perform SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log loss
            loss_history.append(loss.item())
        		
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"batch [{i+1}/{num_batches}], Loss: {loss.item():.4f}")
                    
            if (i+1) % 1000 == 0:
                with open("loss.json", "w") as ofile:
                    json.dump(loss_history, ofile)
            
    return model

def main():
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output-path", default="lenet5-mnist.onnx")
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--random-seed", default=0, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    
    from lenet5mnist.data import get_datasets
    from lenet5mnist.model import LeNet5
    
    # Instantiate model, dataset, loss function, optimizer
    model = LeNet5()
    loss_fct = torch.nn.CrossEntropyLoss()
    train_set, test_set = get_datasets(batch_size=args.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Run training loop
    model = train_model(model, train_set, loss_fct, optimizer, args.num_epochs)

    # Save model
    torch.onnx.export(model, (torch.empty(size=[1,1,32,32])), args.output_path)
