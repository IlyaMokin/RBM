using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.Inizialize
{
	public class NeuronInfo
	{
		public ActivationFunctionEnum ActivationFunction = ActivationFunctionEnum.None;

		public double[] Parameters = null;
		public double[] InputWeights = null;
	}
}
