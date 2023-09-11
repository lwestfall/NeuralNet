namespace NeuralNet.Core.Networks;

using MathNet.Numerics.LinearAlgebra;

public class NetworkParameters
{
    public Vector<float>[] Biases { get; internal set; }

    public Matrix<float>[] Weights { get; internal set; }

    public NetworkParameters(Vector<float>[] biases, Matrix<float>[] weights)
    {
        this.Biases = biases;
        this.Weights = weights;
    }
}
