using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class Sin : ActivationFunction
	{
		private double _minStep = 1e-16;
		private Neuron _neuron;
		private int paramID;
		public Sin(Neuron neuron)
		{
			_neuron = neuron;
			paramID = _neuron.Parameters.Count;
			_neuron.Parameters.Add(1);
			neuron.NeuronStimulus += (alpha) =>
			{
				var s = neuron.S;
				neuron.Parameters[paramID] -= (double)((decimal)alpha * (decimal)DFA(s) * neuron.Error);
			};
		}
		public override double F(double s)
		{
			var val = _neuron.Parameters[paramID] * s;
			return
				val < Math.PI && val > 0 ? Math.Sin(val)
				: val < 0 ? -_minStep
				: _minStep;
		}

		public override double DF(double s)
		{
			var val = _neuron.Parameters[paramID] * s;
			return
				val < Math.PI && val > 0 ? s * Math.Cos(val)
				: val < 0 ? 1
				: -1;
		}

		private double DFA(double s)
		{
			var aRate = _neuron.Parameters[paramID];
			var val = aRate * s;
			return
				val < Math.PI && val > 0 ? aRate * Math.Cos(aRate * s)
				: val < 0 ? 1
				: -1;
		}
	}
}
