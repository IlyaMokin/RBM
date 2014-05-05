using NetworkM;
using NetworkM.ActivationFunctions;
using NetworkM.Networks;
using NetworkM.SaveAndInizialize;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorWithoutHiddenLayer
{
	class Program
	{
		static void Main(string[] args)
		{
			/*xor*/
			var input = new[]{
				new double[]{0,0},
				new double[]{0,1},
				new double[]{1,0},
				new double[]{1,1}
			};
			var output = new[]{
				new double[]{0},
				new double[]{1},
				new double[]{1},
				new double[]{0}
			};

			var network = new NeuralNetwork(
				new SimpleLayerInfo() { CountNeuronsInLayer = 2 },
				new SimpleLayerInfo() { CountNeuronsInLayer = 1, ActivationFunction = ActivationFunctionEnum.Sin }
			);

			var teacher = new NetworkM.Teachers.Backpropagation.GradientDescent(network);
			teacher.Alpha = 0.0025;
			double err = 0;
			do
			{
				err = teacher.RunEpoch(input, output);
				if (teacher.IterationCounter % 10000 == 0)
				{
					Console.WriteLine(err);
				}
			} while (err > 0.001);
			Console.WriteLine(network.GetAbsoluteError(input,output));
			Console.WriteLine();
		}
	}
}
