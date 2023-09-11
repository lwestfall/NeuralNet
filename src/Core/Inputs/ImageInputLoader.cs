namespace NeuralNet.Core.Inputs;

using System.Globalization;
using MathNet.Numerics.LinearAlgebra;

public class DigitInputLoader : InputLoader
{
    public string ImageDirectory { get; set; }

    public Action<IImageProcessingContext> ImageProcessingOperation { get; set; }

    public LabeledData? LabeledData { get; set; }

    public DigitInputLoader(string imageDirectory, Action<IImageProcessingContext> operation)
    {
        this.ImageDirectory = imageDirectory;
        this.ImageProcessingOperation = operation;
    }

    public (LabeledData, LabeledData) Split(float ratio)
    {
        if (this.LabeledData is null)
        {
            throw new InvalidOperationException("Labeled data must be loaded before splitting");
        }

        var n = this.LabeledData.Count;
        var splitIndex = (int)(n * ratio);

        var shuffled = this.LabeledData.ShuffleInPlace();
        var trainingData = new LabeledData(shuffled.InputActivations.Take(splitIndex).ToArray(), shuffled.OutputActivations.Take(splitIndex).ToArray());
        var testData = new LabeledData(shuffled.InputActivations.Skip(splitIndex).ToArray(), shuffled.OutputActivations.Skip(splitIndex).ToArray());

        return (trainingData, testData);
    }

    public override async Task LoadLabeledData(CancellationToken cancellationToken)
    {
        var files = Directory.GetFiles(this.ImageDirectory, "*.jpg", SearchOption.AllDirectories);

        var xs = new List<Vector<float>>(files.Length);
        var ys = new List<Vector<float>>(files.Length);

        foreach (var file in files)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                this.LabeledData = new LabeledData(xs.ToArray(), ys.ToArray());
                return;
            }

            var (x, y) = await this.GetXYForImage(file, cancellationToken);
            xs.Add(x);
            ys.Add(y);
        }

        this.LabeledData = new(xs.ToArray(), ys.ToArray());
    }

    private async Task<(Vector<float>, Vector<float>)> GetXYForImage(string filePath, CancellationToken cancellationToken)
    {
        using var image = await Image.LoadAsync<Rgb24>(filePath, cancellationToken);
        var inputActivations = new float[image.Width * image.Height];
        image.Mutate(this.ImageProcessingOperation);

        for (var yPx = 0; yPx < image.Height; yPx++)
        {
            for (var xPx = 0; xPx < image.Width; xPx++)
            {
                var pixel = image[xPx, yPx];
                var activation = (pixel.R + pixel.G + pixel.B) / 3f / 255f;
                inputActivations[(yPx * image.Width) + xPx] = activation;
            }
        }

        var parentDirName = Path.GetFileName(Path.GetDirectoryName(filePath)!)!;
        var label = int.Parse(parentDirName, CultureInfo.InvariantCulture);

        var x = Vector<float>.Build.Dense(inputActivations);
        var y = Vector<float>.Build.Dense(10);
        y[label] = 1;

        return (x, y);
    }
}
