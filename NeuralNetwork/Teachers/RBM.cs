using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Teachers
{
	public class RBM : GradientDescent
	{
		public RBM(Network network)
			: base(network) { }

		public double RunEpoch(
			IEnumerable<double[]> inputs,
			IEnumerable<double[]> outputs)
		{
			base.RunEpoch(inputs, outputs, false, -1, 1);
			return base.RunEpoch(inputs, outputs, false, -1, 0);
		}

		public override double RunEpoch(
			IEnumerable<double[]> inputs, 
			IEnumerable<double[]> outputs, 
			bool optimize = false, 
			int teachLayerIndex = -1, 
			int from = 1)
		{
			for (int i = 0; i < inputs.Count(); i++)
			{
				double[] result = _network.GetResult(inputs.ElementAt(i), 1);
				if (GetErrorForElement(result, outputs.ElementAt(i)) > Threshold)
				{
					CalculateError(outputs.ElementAt(i), from);
					for (int layerIndex = _layers.Count - 1; layerIndex > 0; layerIndex--)
					{
						foreach (var neuron in _layers[layerIndex].Neurons)
						{
							neuron.Parameters[0] += Alpha * neuron.Error * neuron.Deriviations[0](neuron.S);
							neuron.InputConnections.ForEach(inpLink =>
							{
								inpLink.Parameters[0] -= Alpha * neuron.Error * neuron.Deriviations[0](neuron.S) * inpLink.Neuron.Out;
							});
							neuron.NeuronStimulus(Alpha);
						}
						if (optimize)
						{
							result = _network.GetResult(inputs.ElementAt(i));
							CalculateError(outputs.ElementAt(i));
						}

					}
				}
				/**/
				result = _network.GetResult(_layers[_layers.Count - 1].Neurons.Select(x => x.Out).ToArray(), 0);
				if (GetErrorForElement(result, outputs.ElementAt(i)) > Threshold)
				{
					CalculateError(outputs.ElementAt(i), from);
					for (int layerIndex = _layers.Count - 1; layerIndex > 0; layerIndex--)
					{
						foreach (var neuron in _layers[layerIndex].Neurons)
						{
							neuron.Parameters[0] += Alpha * neuron.Error * neuron.Deriviations[0](neuron.S);
							neuron.InputConnections.ForEach(inpLink =>
							{
								inpLink.Parameters[0] += (neuron.Out - neuron.LastValue) * neuron.Deriviations[0](neuron.S) * inpLink.Neuron.LastValue;
							});
							neuron.NeuronStimulus(Alpha);
						}
						if (optimize)
						{
							result = _network.GetResult(inputs.ElementAt(i));
							CalculateError(outputs.ElementAt(i));
						}

					}
				}
			}
			Error = _network.GetAbsoluteError(inputs, outputs, Threshold);
			IterationCounter += 1;
			return Error;
		}
	}
}
