﻿using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using NeuralNetwork.NetworkComponents;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class Network
	{

		public Network(params LayerInfo[] layers)
		{
			var functionsFactory = new FunctionsFactory();
			var random = new Random(DateTime.Now.Millisecond);
			for (var layerID = 0; layerID < layers.Length; layerID += 1)
			{
				var layer = layers[layerID];
				var nLayer = new Layer();
				Layers.Add(nLayer);

				for (var neuronID = 0; neuronID < layer.CountNeuronsInLayer; neuronID += 1)
				{

					var nNeuron = new Neuron();
					nLayer.Neurons.Add(nNeuron);
					var func = functionsFactory.GetActivationFunction(layer.ActivationFunction, nNeuron);
					nNeuron.ActivationF = func.F;


					nNeuron.Parameters.Add(random.NextDouble() / 2);
					nNeuron.Deriviations = new List<FunctionActivator> { func.DF, func.DDF };

					if (layerID == 0) continue;
					foreach (var prevLayerNeuron in Layers[layerID - 1].Neurons)
					{
						var backConnection = new Connection { Neuron = prevLayerNeuron, Parameters = new List<double> { random.NextDouble() / 2 } };
						var connection = new Connection { Neuron = nNeuron, Parameters = backConnection.Parameters };

						nNeuron.InputConnections.Add(backConnection);
						prevLayerNeuron.OutputConnections.Add(connection);
					}

					if (layer.Recurent != -1)
					{
						foreach (var nextLayerNeuron in Layers[layer.Recurent].Neurons)
						{
							var nextConnection = new Connection { Neuron = nextLayerNeuron, Parameters = new List<double> { random.NextDouble() / 2 } };
							var backconnection = new Connection { Neuron = nNeuron, Parameters = nextConnection.Parameters };

							nNeuron.OutputConnections.Add(nextConnection);
							nextLayerNeuron.InputConnections.Add(backconnection);
						}
					}
				}
			}
		}

		internal IList<Layer> Layers = new List<Layer>();

		public double[] GetResult(double[] inputs, int from = 1)
		{
			for (int i = 0; i < inputs.Length; i++)
			{
				Layers[0].Neurons[i].Out = inputs[i];
			}
			for (int i = from; i < Layers.Count; i++)
			{
				foreach (var neuron in Layers[i].Neurons)
				{
					neuron.S = neuron.InputConnections.Sum(x => x.Parameters[0] * x.Neuron.Out) + neuron.Parameters[0];
					neuron.Out = neuron.ActivationF(neuron.S);
				}
			}
			return Layers.Last().Neurons.Select(x => x.Out).ToArray();
		}

		public double[] GetResultBack()
		{
			for (int i = Layers.Count - 2; i > -1; i--)
			{
				foreach (var neuron in Layers[i].Neurons)
				{
					neuron.S = neuron.OutputConnections.Sum(x => x.Parameters[0] * x.Neuron.Out) + neuron.Parameters[0];
					neuron.Out = neuron.ActivationF(neuron.S);
				}
			}

			return Layers.First().Neurons.Select(x => x.Out).ToArray();
		}

		private static object GetJson(string path)
		{
			using (StreamReader re = new StreamReader(path))
			{
				JsonTextReader reader = new JsonTextReader(re);
				JsonSerializer se = new JsonSerializer();
				return se.Deserialize(reader);
			}
		}

		public double GetAbsoluteError(IEnumerable<double[]> inputs, IEnumerable<double[]> outputs, double threshold = 0, int from = 1)
		{
			double sum = 0.0;
			var outp = outputs.ToList();
			for (int i = 0; i < outputs.Count(); i++)
			{
				double[] res = GetResult(inputs.ElementAt(i), from);

				if (double.NaN.Equals(res[0]))
				{
					throw new ArgumentOutOfRangeException();
				}

				sum += res.Zip(outp[i], (x, y) =>
				{
					double val = Math.Abs(x - y);
					return val > threshold ? val : 0;
				}).Sum();
			}
			return sum;
		}

		public static Network Inizialize(string path)
		{
			var jsonObj = GetJson(path);
			var isRecurentField = (jsonObj as JObject)["isRecurrentStyle"];
			var isRecurent = isRecurentField == null ? false : (bool)isRecurentField;
			var inizializer = new NetworkInizializer();
			if (!isRecurent)
			{
				return inizializer.Simple(jsonObj);
			}
			else
			{
				return inizializer.Recurent(jsonObj);
			}
		}

		public void Serialize(string path)
		{
			var inizializer = new NetworkInizializer();
			using (var writer = new StreamWriter(path))
			{
				writer.WriteLine(inizializer.Serialize(this));
			}
		}
	}
}
