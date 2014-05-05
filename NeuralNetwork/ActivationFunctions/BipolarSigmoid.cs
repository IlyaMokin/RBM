using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class BipolarSigmoid : ActivationFunction
	{
		public override double F(double s)
		{
			return 2.0 / (1.0 + Math.Exp(-s)) - 1;
		}

		public override double DF(double s)
		{
			return 2.0 * Math.Exp(-s) / Math.Pow(1 + Math.Exp(-s), 2);
		}
	}
}
