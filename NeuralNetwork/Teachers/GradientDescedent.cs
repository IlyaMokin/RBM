using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Teachers
{
	public class GradientDescent
	{
		protected Network _network;
		private IList<Layer> _layers;

		public GradientDescent(Network network)
		{
			_layers = network.Layers;
			this._network = network;
		}

		protected void CalculateError(double[] output)
		{
			for (int i = _layers.Count - 1; i > 0; i--)
			{
				for (int k = 0; k < _layers[i].Neurons.Count; k++)
				{
					if (i == _layers.Count - 1)
					{
						_layers[i].Neurons[k].Error = _layers[i].Neurons[k].Out - output[k];
					}
					else
					{
						_layers[i].Neurons[k].Error =
							_layers[i].Neurons[k].OutputConnections.Sum(x =>
								x.Neuron.Error * x.Parameters[0] * x.Neuron.Deriviations[0](x.Neuron.S));
					}
				}
			}
		}

		public double Alpha = 0.15;
		public int IterationCounter = 0;
		public double Threshold = 0.0;
		protected double Error = 0;

		public virtual double RunEpoch(
			IEnumerable<double[]> inputs, 
			IEnumerable<double[]> outputs, 
			bool optimize = false, 
			int teachLayerIndex = -1)
		{

			for (int i = 0; i < inputs.Count(); i++)
			{
				double[] result = _network.GetResult(inputs.ElementAt(i));
				if (GetErrorForElement(result, outputs.ElementAt(i)) > Threshold)
				{
					CalculateError(outputs.ElementAt(i));
					for (int layerIndex = _layers.Count - 1; layerIndex > 0; layerIndex--)
					{
						if (teachLayerIndex != -1 && teachLayerIndex != layerIndex)
						{
							continue;
						}

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
			}
			Error = _network.GetAbsoluteError(inputs, outputs, Threshold);
			IterationCounter += 1;
			return Error;
		}

		protected double GetErrorForElement(double[] res, double[] output)
		{
			return Math.Sqrt(res.Zip(output, (x, y) =>
			{
				double val = x - y;
				return Math.Abs(val) > Threshold ? Math.Pow(val, 2) : 0;
			}).Sum());
		}
	}
}
