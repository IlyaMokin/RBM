using NeuralNetwork;
using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using NeuralNetwork.Teachers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MnistRbm
{
	class MnistRbm
	{
		static void Main(string[] args)
		{
			var input = new Mnist().Read("train-images.idx3-ubyte")
				.Select(x=>x.Select(y=>y/255).ToArray()).Take(10);

			var network = new Network(
				new LayerInfo { CountNeuronsInLayer = 784 },
				new LayerInfo { CountNeuronsInLayer = 600, ActivationFunction = ActivationFunctionEnum.Sigmoid },
				new LayerInfo { CountNeuronsInLayer = 784, ActivationFunction = ActivationFunctionEnum.Liner }
			);


			GradientDescent teacher = new RBM(network);
			teacher.Alpha = 0.25;
			double
				err = 0,
				lastErr = 100,
				step = 100;

			int
				counter = 0;

			for (int i = 0; i < 2; i += 1)
			{
				err = teacher.RunEpoch(input, input, false);
				lastErr = err;
				if (counter++ % 1 == 0) Console.WriteLine(string.Format("{0:f4} /{1:f4} ", err, network.GetAbsoluteError(input, input)));
			}

			Console.WriteLine("-------"); Console.ReadKey();/**/


			teacher = new GradientDescent(network);
			teacher.Alpha = 0.0025;
			do
			{
				//teacher.Alpha = teacher.Alpha > 0.0001 ? teacher.Alpha / 1.2 : 0.0001;
				do
				{
					err = teacher.RunEpoch(input, input)/input.Count();
					lastErr = err;
					if (counter++ % 1 == 0) Console.WriteLine(string.Format("{0:f4} / {1:f4}//{2:f4}", err, counter, teacher.Alpha));
				} while (err > 0.01 && counter % 200 != 0);

			} while (err > 0.01);
		}
	}
}
