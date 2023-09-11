namespace NeuralNet.Core.Inputs;

public abstract class InputLoader
{
    public abstract Task LoadLabeledData(CancellationToken cancellationToken);
}
