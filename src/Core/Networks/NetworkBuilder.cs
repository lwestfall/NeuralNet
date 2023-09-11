namespace NeuralNet.Core.Networks;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet.Core.Utils;

public class NetworkBuilder
{
    public List<int> Shape { get; set; } = new();

    public Func<Vector<float>, Vector<float>> ActivationFunction { get; set; } = MathUtil.Sigmoid;

    public Func<Vector<float>, Vector<float>> ActivationDerivativeFunction { get; set; } = MathUtil.SigmoidPrime;

    public NetworkBuilder AddLayer(int width)
    {
        this.Shape.Add(width);
        return this;
    }

    public NetworkBuilder ActivateUsing(Func<Vector<float>, Vector<float>> activation, Func<Vector<float>, Vector<float>> derivative)
    {
        this.ActivationFunction = activation;
        this.ActivationDerivativeFunction = derivative;
        return this;
    }

    public Network Build()
    {
        var weights = new List<Matrix<float>>();
        var biases = new List<Vector<float>>();

        var lastLayerSize = this.Shape.First();

        foreach (var layerSize in this.Shape.Skip(1))
        {
            // randomly initialize weights and biases from -1.0 to 1.0
            biases.Add(
                Vector<float>.Build.Dense(
                    layerSize,
                    (_) => (MathUtil.Random.NextSingle() * 2) - 1
                ));

            weights.Add(
                Matrix<float>.Build.Dense(
                    layerSize,
                    lastLayerSize,
                    (_, _) => (MathUtil.Random.NextSingle() * 2) - 1
                ));
            lastLayerSize = layerSize;
        }

        var networkParams = new NetworkParameters(biases.ToArray(), weights.ToArray());

        var network = new Network(networkParams)
        {
            ActivationFunction = this.ActivationFunction,
            ActivationFunctionDerivative = this.ActivationDerivativeFunction
        };

        return network;
    }
}
