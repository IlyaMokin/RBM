using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class Gauss : ActivationFunction
	{

		private Neuron _neuron;
		private int _param1ID;
		private int _param2ID;
		public Gauss(Neuron neuron)
		{
			_neuron = neuron;
			_param1ID = neuron.Parameters.Count;
			neuron.Parameters.Add(0.1);
			_param2ID = neuron.Parameters.Count;
			neuron.Parameters.Add(0.4);
			neuron.NeuronStimulus += (alpha) =>
			{
				var s = neuron.S;
				var dfRate1 = DF_rate1(s);
				var dfRate2 = DF_rate2(s);
				neuron.Parameters[_param1ID] -= alpha * dfRate1 * neuron.Error;
				neuron.Parameters[_param2ID] -= alpha * dfRate2 * neuron.Error;
			};
		}

		private double f(double s, double sigma, double m)
		{
			//rate1  - sigma / rate2  - mu
			return Math.Exp(-Math.Pow(s - m, 2) * 0.5 * Math.Pow(sigma, -2));
		}

		public override double F(double s)
		{
			return f(s, _neuron.Parameters[_param1ID], _neuron.Parameters[_param2ID]);
		}

		private double DF_rate1(double s)
		{
			double sigma = _neuron.Parameters[_param1ID];
			double m = _neuron.Parameters[_param2ID];
			double temp = Math.Pow(s - m, 2);

			double c =
				Math.Exp(-temp * 0.5 * Math.Pow(sigma, -2));

			return c * temp  * Math.Pow(sigma, -3);
		}

		private double DF_rate2(double s)
		{
			double sigma = _neuron.Parameters[_param1ID];
			double m = _neuron.Parameters[_param2ID];
			double temp = Math.Pow(s - m, 2);
			double c =
				Math.Exp(-temp * 0.5 * Math.Pow(sigma, -2));
			return c * (s - m) * Math.Pow(sigma, -2);
		}
	}
}
