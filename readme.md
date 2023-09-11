# NeuralNet

This is a basic proof of concept for a neural network implemented in .NET 7. It's heavily based on Michael Nielsen's excellent book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), with some minor twists to make it a little more object oriented.

The purpose of this project is simply to get more hands on experience with how neural networks work.

## Setup

I'm assuming you have .NET 7 installed. If not, you can get it [here](https://dotnet.microsoft.com/download/dotnet/7.0).

1. Clone this repo
2. Download the MNIST dataset from [here](https://www.kaggle.com/datasets/scolianni/mnistasjpg/download?datasetVersionNumber=1) and extract it to the `Datasets/digits` folder

## To Run

1. `cd src/Sandbox`
2. `dotnet run`

## What's Next?

- [ ] Add a way to save and load trained networks
- [ ] Implement ReLU activation function
- [ ] Add convolutions
- [ ] Add unit testing
- [ ] Benchmarks and performance optimizations

## References

<http://neuralnetworksanddeeplearning.com/>

<https://github.com/tromgy/simple-neural-networks/blob/master/neural-network.ipynb>

<https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi>
