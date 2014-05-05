using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class Bithreshold : ActivationFunction
	{
		public override double F(double s)
		{
			return 
				s < 0 ? 0 :
				s >= 1 ? 0 : 1;
		}

		public override double DF(double s)
		{
			return 1;
		}
	}
}
