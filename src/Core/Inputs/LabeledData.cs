namespace NeuralNet.Core.Inputs;

using MathNet.Numerics.LinearAlgebra;
using NeuralNet.Core.Utils;

public class LabeledData
{
    public Vector<float>[] InputActivations { get; set; }

    public Vector<float>[] OutputActivations { get; set; }

    public int Count { get; set; }

    public LabeledData(Vector<float>[] inputActivations, Vector<float>[] outputActivations)
    {
        if (inputActivations.Length != outputActivations.Length)
        {
            throw new ArgumentException("Input and output activations must have the same length");
        }

        this.InputActivations = inputActivations;
        this.OutputActivations = outputActivations;
        this.Count = this.InputActivations.Length;
    }

    public static implicit operator LabeledData((Vector<float>[] x, Vector<float>[] y) data) => new(data.x, data.y);

    public LabeledData ShuffleInPlace()
    {
        var n = this.Count;

        while (n > 1)
        {
            n--;
            var k = MathUtil.Random.Next(n + 1);
            var x = this.InputActivations[k];
            var y = this.OutputActivations[k];
            this.InputActivations[k] = this.InputActivations[n];
            this.OutputActivations[k] = this.OutputActivations[n];
            this.InputActivations[n] = x;
            this.OutputActivations[n] = y;
        }

        return this;
    }

    public Vector<float>[][] Zip()
    {
        var zipped = new Vector<float>[this.Count][];

        for (var i = 0; i < this.Count; i++)
        {
            zipped[i] = new[] { this.InputActivations[i], this.OutputActivations[i] };
        }

        return zipped;
    }
}
