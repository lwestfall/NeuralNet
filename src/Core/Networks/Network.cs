namespace NeuralNet.Core.Networks;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet.Core.Inputs;
using NeuralNet.Core.Utils;

public class Network
{
    public NetworkParameters Parameters { get; private set; }

    public Network(NetworkParameters parameters) => this.Parameters = parameters;

    public Func<Vector<float>, Vector<float>> ActivationFunction { get; set; } = MathUtil.Sigmoid;

    public Func<Vector<float>, Vector<float>> ActivationFunctionDerivative { get; set; } = MathUtil.SigmoidPrime;

    public void Learn(LabeledData trainingData, int epochs, int miniBatchSize, float learningRate, LabeledData? testData = null)
    {
        var n = trainingData.Count;

        for (var j = 0; j < epochs; j++)
        {
            trainingData.ShuffleInPlace();

            var zippedData = trainingData.Zip();
            var miniBatches = zippedData.Chunk(miniBatchSize);

            foreach (var miniBatch in miniBatches)
            {
                this.StochasticGradientDescent(miniBatch, learningRate);
            }

            if (testData != null)
            {
                var accuracy = 100.0 * this.Evaluate(testData) / testData.Count;
                Console.WriteLine($"Epoch {j}: accuracy {accuracy:F2}%");
            }
            else
            {
                Console.WriteLine($"Epoch {j} complete");
            }
        }
    }

    private void StochasticGradientDescent(Vector<float>[][] miniBatch, float eta)
    {
        var nabla_b = new List<Vector<float>>();
        var nabla_w = new List<Matrix<float>>();

        for (var i = 0; i < this.Parameters.Biases.Length; i++)
        {
            nabla_b.Add(Vector<float>.Build.Dense(this.Parameters.Biases[i].Count));
        }

        for (var i = 0; i < this.Parameters.Weights.Length; i++)
        {
            nabla_w.Add(Matrix<float>.Build.Dense(this.Parameters.Weights[i].RowCount, this.Parameters.Weights[i].ColumnCount));
        }

        for (var i = 0; i < miniBatch.Length; i++)
        {
            var x = miniBatch[i][0];
            var y = miniBatch[i][1];

            var gradientDeltas = this.BackPropagate(x, y);
            nabla_b = nabla_b.Zip(gradientDeltas.Biases, (nb, dnb) => nb + dnb).ToList();
            nabla_w = nabla_w.Zip(gradientDeltas.Weights, (nw, dnw) => nw + dnw).ToList();
        }

        // Suppress warning "Remove unnecessary parentheses" - I think this is more readable
#pragma warning disable IDE0047
        this.Parameters.Weights = this.Parameters.Weights.Zip(nabla_w, (w, nw) => w - ((eta / miniBatch.Length) * nw)).ToArray();
        this.Parameters.Biases = this.Parameters.Biases.Zip(nabla_b, (b, nb) => b - ((eta / miniBatch.Length) * nb)).ToArray();
#pragma warning restore IDE0047
    }

    // roughly as written by Nielsen, adapted for C#
    public NetworkParameters BackPropagate(Vector<float> x, Vector<float> y)
    {
        // feedforward
        var activation = x;
        var activations = new List<Vector<float>> { x };
        var zs = new List<Vector<float>>();

        for (var i = 0; i < this.Parameters.Weights.Length; i++)
        {
            var z = (this.Parameters.Weights[i] * activation) + this.Parameters.Biases[i];
            zs.Add(z);
            activation = this.ActivationFunction(z);
            activations.Add(activation);
        }

        // backward pass
        var nabla_b = new Vector<float>[this.Parameters.Biases.Length];
        var nabla_w = new Matrix<float>[this.Parameters.Weights.Length];

        var delta = CostDerivative(activations[^1], y).PointwiseMultiply(this.ActivationFunctionDerivative(zs[^1]));

        nabla_b[^1] = delta;
        nabla_w[^1] = delta.ToColumnMatrix() * activations[^2].ToRowMatrix();

        for (var i = 2; i <= this.Parameters.Weights.Length; i++)
        {
            var z = zs[^i];
            var actPrime = this.ActivationFunctionDerivative(z);

            delta = this.Parameters.Weights[^(i - 1)].Transpose() * delta;
            delta = delta.PointwiseMultiply(actPrime);
            nabla_b[^i] = delta;

            nabla_w[^i] = delta.ToColumnMatrix() * activations[^(i + 1)].ToRowMatrix();
        }

        return new(nabla_b, nabla_w);
    }

    public Vector<float> FeedForward(Vector<float> a)
    {
        foreach (var (b, w) in this.Parameters.Biases.Zip(this.Parameters.Weights, (b, w) => (b, w)))
        {
            a = this.ActivationFunction((w * a) + b);
        }

        return a;
    }

    public int GetActivatedOutputIndex(Vector<float> input) => this.FeedForward(input).MaximumIndex();

    public int Evaluate(LabeledData testData)
    {
        var testResults = testData.InputActivations.Select(this.GetActivatedOutputIndex).ToArray();
        var expectedResults = testData.OutputActivations.Select(x => x.MaximumIndex()).ToArray();

        return testResults.Zip(expectedResults, (x, y) => x == y).Count(x => x);
    }

    public static Vector<float> CostDerivative(Vector<float> outputActivations, Vector<float> y) => outputActivations - y;
}

