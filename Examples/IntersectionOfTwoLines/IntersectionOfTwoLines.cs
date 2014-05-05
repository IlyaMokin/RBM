using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using NeuralNetwork;

namespace IntersectionOfTwoLines
{
	class IntersectionOfTwoLines
	{
		static void Main(string[] args)
		{
			/*var network = new Network(
				new LayerInfo() { CountNeuronsInLayer = 2 },
				new StrictLayerInfo()
				{
					Neurons = new[] { 
						new NeuronInfo() { ActivationFunction = ActivationFunctionEnum.Threshold,T=0,InputWeights = new double[]{-1, 1 } },
						new NeuronInfo() { ActivationFunction = ActivationFunctionEnum.Threshold,T=0,InputWeights = new double[]{1, 1 } } 
						}
				},
				new StrictLayerInfo()
				{
					Neurons = new[] { 
						new NeuronInfo() { ActivationFunction = ActivationFunctionEnum.Bithreshold,T=0.5,InputWeights = new double[]{1, 1 }}}
				}
			);

			var rand = new Random();
			var input = new List<double[]>();
			var output = new List<double[]>();
			for (var i = 0; i < 50; i += 1)
			{
				var inp = new[] { rand.NextDouble()*10 * 2 - 10, rand.NextDouble()*10 * 2 - 10 };
				input.Add(inp);
				output.Add(network.GetResult(inp));
			}


			network = new NeuralNetwork(
				new SimpleLayerInfo() { CountNeuronsInLayer = 2 },
				new SimpleLayerInfo() { CountNeuronsInLayer = 4 , ActivationFunction = ActivationFunctionEnum.GiperbalTan },
				new SimpleLayerInfo() { CountNeuronsInLayer = 1, ActivationFunction = ActivationFunctionEnum.Sin }
			);

			//network = NeuralNetwork.Inizialize("output.txt");
			var teacher = new NetworkM.Teachers.Backpropagation.GradientDescent(network);
			teacher.Alpha = 0.025;
			double err = 0;
			do
			{
				err = teacher.RunEpoch(input, output,true);
				if (teacher.IterationCounter % 100 == 0)
				{
					Console.WriteLine(err);
				}
			} while (err > 0.1);/**/

			/*Thread.CurrentThread.CurrentCulture = System.Globalization.CultureInfo.GetCultureInfo("EN-US");
			while(true){
				var inp  = Console.ReadLine().Split(' ').Select(x=>double.Parse(x)).ToArray();
				Console.WriteLine(Math.Round(network.GetResult(inp)[0],4));
				Console.WriteLine();
			}*/
			//network.Serialize("output.txt");*/
		}
	}
}
