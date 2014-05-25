using NeuralNetwork;
using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using NeuralNetwork.Teachers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace XorWithOneHiddenLayer
{
	class Xor
	{
		static void Main(string[] args)
		{

			int w = 2; //--кол-во элементов множества
			var inputs = new object[(int)Math.Pow(2d, (double)w)]
				.Select((o, maskIndex) => new double[w]//Перебор возможных масок
					.Select((val, index) => (maskIndex & (1 << index)) != 0 ? 1d : 0d).ToArray())
				.ToArray();
			var outputs = inputs.Select(inp => inp.Aggregate((x, y) => (double)((int)x ^ (int)y)))
				.Select(x => new[] { x })
				.ToArray();

			var network = new Network(
				new LayerInfo { CountNeuronsInLayer = w },
				new LayerInfo { CountNeuronsInLayer = 2, ActivationFunction = ActivationFunctionEnum.Gauss },
				new LayerInfo { CountNeuronsInLayer = 1, ActivationFunction = ActivationFunctionEnum.Gauss });

			var teacher = new GradientDescent(network);
			using (var log = new CsvStatistic("../../Stat/Xor_Gauss_2l.csv", false))
			{
				Teach(inputs, outputs, teacher, log);
			}
			network.Serialize("../../Stat/Gauss_2layer.json");
		}

		private static void Teach(double[][] inputs, double[][] outputs, GradientDescent teacher, CsvStatistic log)
		{
			teacher.Alpha = 0.25;

			double
				err = 0,
				lastErr = 100;

			int
				counter = 0;

			do
			{
				for (var id = 0; id < 3; id += 1)
				{
					do
					{
						err = teacher.RunEpoch(inputs, outputs, false, id);
						lastErr = err;
						if (counter++ % 10000 == 0) Console.WriteLine(
						string.Format("err {0:f4} / step {1:f6} /index {2} / alpha {3:f4}",
									err, counter, id, teacher.Alpha));
					} while (err > 0.01 && counter % 50000 != 0);
				}

				do
				{
					err = teacher.RunEpoch(inputs, outputs);
					lastErr = err;

					if (counter % 1000 == 0)

						log.Write(err, counter, teacher.Alpha);


					if (counter++ % 1000 == 0)
						Console.WriteLine(string.Format("err {0:f4} / step {1:f6}/ aplha{2:f4}", err, counter, teacher.Alpha));

				} while (err > 0.01 && counter % 50000 != 0);

				log.Write(err, counter, teacher.Alpha);
				teacher.Alpha = teacher.Alpha > 0.0001 ? teacher.Alpha / 1.2 : 0.0001;
			} while (err > 0.01);
		}
	}
}
