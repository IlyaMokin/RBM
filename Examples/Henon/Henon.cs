using NeuralNetwork;
using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.Inizialize;
using NeuralNetwork.Teachers;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Henon
{
	class Henon
	{
		static IEnumerable<double> GetHenonList(int number)
		{
			var list = new List<double>(500);
			CultureInfo culture = new CultureInfo("en-us");
			int counter = 0;
			using (var reader = new StreamReader("HENON.TXT"))
			{
				while (!reader.EndOfStream && number > counter++)
				{
					list.Add(double.Parse(reader.ReadLine(), culture));
				}
			}
			return list;
		}

		static void Main(string[] args)
		{
			var henon = GetHenonList(3000);
			var helpValue = 0;
			var min = henon.Min();
			henon = henon.Select(x => x + Math.Abs(min));
			var max = henon.Max();

			henon = henon.Select(x => x / max);/**/

			double[][] input = henon
						.Select((h, index) => new { groupValue = index % 8 != 0 || index == 0 ? helpValue : ++helpValue, value = h })
						.GroupBy(v => v.groupValue, (key, values) => values.Select(x => x.value).ToArray())
						.Take(374)
						.ToArray();

			double[][] output = henon
						.Skip(8)
						.Where((x, index) => index % 8 == 0)
						.Select(x => new[] { x }).ToArray();

			var network = new Network(
				new LayerInfo { CountNeuronsInLayer = 8, ActivationFunction = ActivationFunctionEnum.Gauss },
				new LayerInfo { CountNeuronsInLayer = 5, ActivationFunction = ActivationFunctionEnum.Gauss },
				new LayerInfo { CountNeuronsInLayer = 1, ActivationFunction = ActivationFunctionEnum.Liner }
			);/**/
			//var network = Network.Inizialize(@"E:\Code\NeuralNetworkModern\NeuralNetwork\JsonInitExample\RBM_8_5_1.json");
			//network.Serialize("RBM_8_5_1.json");
			

			var teacher = new GradientDescent(network);
			teacher.Alpha = 0.025;
			double
				err = 0,
				lastErr = 100,
				step = 100;

			int
				counter = 0;

			do
			{
				teacher.Alpha = teacher.Alpha > 0.0001 ? teacher.Alpha / 1.2 : 0.0001;

				for (var id = 0; id < 3; id += 1)
				{
					do
					{
						err = teacher.RunEpoch(input, output, false, id);
						lastErr = err;
						if (counter++ % 100 == 0) Console.WriteLine(string.Format("{0} / {1} / {2} / {3}", err, counter, id, teacher.Alpha));
					} while (err > 0.01 && counter % 200 != 0);
				}

				do
				{
					err = teacher.RunEpoch(input, output);
					lastErr = err;
					if (counter++ % 100 == 0) Console.WriteLine(string.Format("{0:f4} / {1:f4}//{2:f4}", err, counter, teacher.Alpha));
				} while (err > 0.01 && counter % 200 != 0);

			} while (err > 0.01);
		}
	}
}
