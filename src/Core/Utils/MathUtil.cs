namespace NeuralNet.Core.Utils;

using MathNet.Numerics.LinearAlgebra;

public static class MathUtil
{
    public static Random Random { get; set; } = new();

    private static float Sigmoid(float x) => 1 / (1 + MathF.Exp(-x));

    private static float SigmoidPrime(float x) => Sigmoid(x) * (1 - Sigmoid(x));

    // https://stats.stackexchange.com/questions/367057/bad-performance-with-relu-activation-function-on-mnist-data-set
    private static float ReLU(float x) => MathF.Max(0, x);

    private static float ReLUPrime(float x) => x > 0 ? 1 : 0;

    public static Vector<float> Sigmoid(Vector<float> x)
    {
        return Vector<float>.Build.DenseOfArray(
            x.Select(Sigmoid).ToArray()
        );
    }

    public static Vector<float> SigmoidPrime(Vector<float> x)
    {
        return Vector<float>.Build.DenseOfArray(
            x.Select(SigmoidPrime).ToArray()
        );
    }

    public static Vector<float> ReLU(Vector<float> x)
    {
        return Vector<float>.Build.DenseOfArray(
            x.Select(ReLU).ToArray()
        );
    }

    public static Vector<float> ReLUPrime(Vector<float> x)
    {
        return Vector<float>.Build.DenseOfArray(
            x.Select(ReLUPrime).ToArray()
        );
    }
}
