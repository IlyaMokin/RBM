using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.NetworkComponents;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Inizialize
{
	class NetworkInizializer
	{
		public Network Simple(object jsonObj)
		{
			var network = new Network();
			var layers = ((jsonObj as JObject)["layers"] as JArray);
			var layerID = -1;
			var functionsFactory = new FunctionsFactory();

			foreach (var layer in layers)
			{
				var networkLayer = new Layer();
				var neurons = layer["neurons"] as JArray;
				foreach (var neuron in neurons)
				{
					var networkNeuron = new Neuron();
					networkNeuron.Parameters = JsonConvert.DeserializeObject<double[]>(neuron["parameters"].ToString()).ToList();

					var activationFunc = functionsFactory.GetActivationFunction((string)neuron["activator"], networkNeuron);
					networkNeuron.ActivationF = activationFunc.F;
					networkNeuron.Deriviations.Add(activationFunc.DF);
					networkNeuron.Deriviations.Add(activationFunc.DDF);

					if (neuron["inputs"] != null)
					{
						var inputs = neuron["inputs"] as JArray;
						foreach (var input in inputs)
						{
							var networkInput = new Connection();
							var networkBackConnection = new Connection();
							var neuronForConnectionID = (int)input["connection"]["neuronID"];
							var neuronForConnection = network.Layers[layerID].Neurons[neuronForConnectionID];

							networkInput.Parameters = JsonConvert.DeserializeObject<double[]>(input["parameters"].ToString());

							networkInput.Neuron = neuronForConnection;

							networkBackConnection.Parameters = networkInput.Parameters;
							networkBackConnection.Neuron = networkNeuron;

							neuronForConnection.OutputConnections.Add(networkBackConnection);
							networkNeuron.InputConnections.Add(networkInput);
						}
					}
					networkLayer.Neurons.Add(networkNeuron);
				}
				network.Layers.Add(networkLayer);
				layerID += 1;
			}

			return network;
		}

		public Network Recurent(object jsonObj)
		{
			var network = Simple(jsonObj);
			var connections = ((jsonObj as JObject)["connections"] as JArray);

			foreach (var jsonConnection in connections)
			{
				var from = new
				{
					layerID = (int)jsonConnection["from"]["layerID"],
					neuronID = (int)jsonConnection["from"]["neuronID"]
				};
				var to = new
				{
					layerID = (int)jsonConnection["to"]["layerID"],
					neuronID = (int)jsonConnection["to"]["neuronID"]
				};
				var parameters = JsonConvert.DeserializeObject<double[]>(jsonConnection["parameters"].ToString());

				Neuron
					fromNeuron = network.Layers[from.layerID].Neurons[from.neuronID],
					toNeuron = network.Layers[to.layerID].Neurons[to.neuronID];

				Connection
					connection = new Connection { Neuron = toNeuron, Parameters = parameters.ToList() },
					backConnection = new Connection { Neuron = fromNeuron, Parameters = parameters.ToList() };

				fromNeuron.OutputConnections.Add(connection);
				toNeuron.InputConnections.Add(backConnection);
			}

			return network;
		}

		public string Serialize(Network network)
		{
			var jNetwork = JObject.Parse("{ isRecurrentStyle: true }");
			var jConnections = new JArray();
			var jLayers = new JArray();

			var layerID = 0;
			foreach (var layer in network.Layers)
			{
				var neuronID = 0;
				var jLayer = new JObject();
				var jNeurons = new JArray();
				jLayer["neurons"] = jNeurons;

				foreach (var neuron in layer.Neurons)
				{
					var jNeuron = new JObject();
					jNeuron["activator"] = neuron.ActivationF.Method.ReflectedType.Name;
					jNeuron["parameters"] = new JArray(neuron.Parameters);
					var inpIndex = 0;
					foreach (var inpConnection in neuron.InputConnections)
					{
						var jConnection = new JObject();

						jConnection["parameters"] = new JArray(inpConnection.Parameters);
						var indexes = GetIndexes(network.Layers, inpConnection.Neuron);

						jConnection["to"] = new JObject();
						jConnection["to"]["layerID"] = layerID;
						jConnection["to"]["neuronID"] = neuronID;

						jConnection["from"] = new JObject();
						jConnection["from"]["layerID"] = indexes[0];
						jConnection["from"]["neuronID"] = indexes[1];

						jConnections.Add(jConnection);
						inpIndex += 1;
					}
					jNeurons.Add(jNeuron);
					neuronID += 1;
				}
				jLayers.Add(jLayer);
				layerID += 1;
			}
			jNetwork["layers"] = jLayers;
			jNetwork["connections"] = jConnections;
			return jNetwork.ToString();
		}

		private int[] GetIndexes(IList<Layer> layers, Neuron neuronForSearch)
		{
			for (var layerID = 0; layerID < layers.Count; layerID += 1)
			{
				var neuronID = layers[layerID].Neurons.IndexOf(neuronForSearch);
				if (neuronID > -1)
				{
					return new int[] { layerID, neuronID };
				}
			}
			return new int[0];
		}
	}
}
