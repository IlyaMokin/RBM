using NeuralNetwork;
using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using NeuralNetwork.Teachers;
using System;

namespace Simple
{
	class Simple
	{
		static void Main(string[] args)
		{
			/*And*/
			var input = new[]{
				new double[]{0,0},
				new double[]{0,1},
				new double[]{1,0},
				new double[]{1,1}
			};
			var output = new[]{
				new double[]{0},
				new double[]{0},
				new double[]{0},
				new double[]{1}
			};

			var network = new Network(
				new LayerInfo() { CountNeuronsInLayer = 2 },
				new LayerInfo() { CountNeuronsInLayer = 1, ActivationFunction = ActivationFunctionEnum.Sigmoid });
			//var network = Network.Inizialize("result.json");
			var teacher = new GradientDescent(network);
			teacher.Alpha = 0.25;
			double err = 0;
			do
			{
				err = teacher.RunEpoch(input, output);
				if (teacher.IterationCounter % 1000 == 0)
				{
					Console.WriteLine(err);
				}
			} while (err > 0.1);
			network.Serialize("result.json");
			Console.WriteLine(err);
		}
	}
}
