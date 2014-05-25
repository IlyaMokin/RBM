using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Inizialize
{
	public class LayerInfo
	{
		public int CountNeuronsInLayer { get; set; }
		public ActivationFunctionEnum ActivationFunction = ActivationFunctionEnum.None;
		public int Recurent = -1;

		IEnumerable<NeuronInfo> Neurons
		{
			get
			{
				for (int i = 0; i < CountNeuronsInLayer; i += 1)
				{
					yield return new NeuronInfo() { ActivationFunction = this.ActivationFunction };
				}
			}
		}
	}
}
