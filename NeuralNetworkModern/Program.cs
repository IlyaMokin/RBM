using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkModern
{
	class Program
	{
		static void Main(string[] args)
		{
			//var network = NeuralNetwork.Network.Inizialize(@"test.json");
			var network = new NeuralNetwork.Network(
				new LayerInfo { CountNeuronsInLayer = 2 },
				new LayerInfo { ActivationFunction = ActivationFunctionEnum.Sigmoid, CountNeuronsInLayer = 2 },
				new LayerInfo { ActivationFunction = ActivationFunctionEnum.Liner, CountNeuronsInLayer = 1 });

			network.Serialize("test.json");/**/

			var t = network.GetResult(new [] {0.1,0.3});
		}
	}
}
