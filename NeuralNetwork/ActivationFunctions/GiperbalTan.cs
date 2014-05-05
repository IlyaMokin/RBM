using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class GiperbalTan : ActivationFunction
	{
		public override double F(double s)
		{
			return Math.Tanh(s);
		}

		public override double DF(double s)
		{
			return 1 - Math.Pow(F(s), 2);
		}
	}
}
