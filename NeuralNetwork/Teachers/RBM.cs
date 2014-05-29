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

		public override double RunEpoch(
			IEnumerable<double[]> inputs,
			IEnumerable<double[]> outputs,
			bool optimize = false,
			int teachLayerIndex = -1,
			int from = 1)
		{
			Error=0;
			for (int i = 0; i < inputs.Count(); i++)
			{
				_network.GetResult(inputs.ElementAt(i));
				/*CalculateError(outputs.ElementAt(i), 1);
				for (int layerIndex = _layers.Count - 1; layerIndex > 0; layerIndex--)
				{
					foreach (var neuron in _layers[layerIndex].Neurons)
					{
						neuron.NeuronStimulus(Alpha);
					}
				}/**/
				_network.GetResultBack();


				for (int layerIndex = _layers.Count - 2; layerIndex > 0; layerIndex--)
				{
					if (!(teachLayerIndex == -1 || teachLayerIndex == layerIndex))
					{
						continue;
					}

					foreach (var neuron in _layers[layerIndex].Neurons)
					{
						neuron.Parameters[0] += Alpha * (neuron.LastValue - neuron.Out);
						neuron.InputConnections.ForEach(inp =>
						{
							inp.Parameters[0] += Alpha * (neuron.LastValue * inp.Neuron.LastValue - neuron.Out * inp.Neuron.Out);
						});
						neuron.NeuronStimulus(Alpha);
					}
				}
				Error += GetSumEnergy();
			}
			
			IterationCounter += 1;
			return Error;
		}

		private double GetSumEnergy()
		{
			var sum = 0d;
			for (int layerIndex = _layers.Count - 2; layerIndex > 0; layerIndex--)
			{

				foreach (var neuron in _layers[layerIndex].Neurons)
				{
					sum += Math.Pow(neuron.LastValue - neuron.Out, 2) / 2;
				}
			}
			return sum;
		}
	}
}
