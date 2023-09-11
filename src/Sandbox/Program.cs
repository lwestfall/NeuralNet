using NeuralNet.Core.Inputs;
using NeuralNet.Core.Networks;
using NeuralNet.Core.Utils;
using SixLabors.ImageSharp.Processing;

// set a seed for reproducibility
// remove / comment the following two lines for non-deterministic behavior between runs
var seed = 1234;
MathUtil.Random = new Random(seed);

// build the neural network
var network = new NetworkBuilder()
    .ActivateUsing(MathUtil.Sigmoid, MathUtil.SigmoidPrime)
    .AddLayer(784)
    .AddLayer(16)
    .AddLayer(16)
    .AddLayer(10)
    .Build();

Console.WriteLine($"Network has {network.Parameters.Biases.Length + 1} layers");

// load the MNIST dataset
Console.WriteLine("Loading MNIST dataset...");
var imageBasePath = @"../../Datasets/digits/trainingSet/trainingSet";
var trainingLoader = new DigitInputLoader(imageBasePath, ctx => ctx.Resize(28, 28).Grayscale());

await trainingLoader.LoadLabeledData(CancellationToken.None);
var (trainingData, testData) = trainingLoader.Split(0.8f);

Console.WriteLine("Loading complete. Beginning training...");

// train the network
network.Learn(trainingData, 5, 10, 3, testData);
