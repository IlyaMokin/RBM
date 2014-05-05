using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace NeuralNetwork.ActivationFunctions.Common
{
	internal class FunctionsFactory
	{
		private static Random _random = new Random(DateTime.Now.Millisecond);
		private Dictionary<ActivationFunctionEnum, ActivationFunction> _activationFunctions = new Dictionary<ActivationFunctionEnum, ActivationFunction>();
		private static IEnumerable<ActivationFunctionEnum> _randomIgnoreList = new[] { 
			ActivationFunctionEnum.GaussStat,
			ActivationFunctionEnum.ExpSin, 
			ActivationFunctionEnum.Bithreshold, 
			ActivationFunctionEnum.Threshold,
			ActivationFunctionEnum.None
		};

		private static IEnumerable<ActivationFunctionEnum> _notCashingList = new[] { 
			ActivationFunctionEnum.GaussStat,
			ActivationFunctionEnum.Sin
		};

		public ActivationFunction GetActivationFunction(ActivationFunctionEnum activationFunction,Neuron neuron)
		{
			if (_notCashingList.Contains(activationFunction))
			{
				return GetNewActivationFunction(activationFunction, neuron);
			}
			else if (!_activationFunctions.Keys.Contains(activationFunction))
			{
				_activationFunctions[activationFunction] = GetNewActivationFunction(activationFunction, neuron);
			}

			return _activationFunctions[activationFunction];
		}

		private ActivationFunction GetNewActivationFunction(ActivationFunctionEnum activationFunction, Neuron neuron)
		{
			switch (activationFunction)
			{
				case ActivationFunctionEnum.BipolarSigmoid:
					return new BipolarSigmoid();
				case ActivationFunctionEnum.GiperbalTan:
					return new GiperbalTan();
				case ActivationFunctionEnum.Sigmoid:
					return new Sigmoid();
				case ActivationFunctionEnum.Sin:
					return new Sin(neuron);
				case ActivationFunctionEnum.Liner:
					return new Liner();
				case ActivationFunctionEnum.Random:
					var functions = Enum.GetValues(typeof(ActivationFunctionEnum)).Cast<ActivationFunctionEnum>()
						.Where(f => !_randomIgnoreList.Contains(f));
					var functionsCount = functions.Count();
					return GetActivationFunction(functions.ElementAt(_random.Next(functionsCount)),neuron);
				case ActivationFunctionEnum.GaussStat:
					return new GaussStat(neuron);
				case ActivationFunctionEnum.Gauss:
					return new Gauss(neuron);
				case ActivationFunctionEnum.ExpSin:
					return new ExpSin();
				case ActivationFunctionEnum.Bithreshold:
					return new Bithreshold();
				case ActivationFunctionEnum.Threshold:
					return new Threshold();
				default:
					return new None();
			}
		}

		public ActivationFunction GetActivationFunction(string name,Neuron neuron)
		{
			return GetActivationFunction(GetActivationFunctionEnum(name), neuron);
		}

		private ActivationFunctionEnum GetActivationFunctionEnum(string activationFunction)
		{
			return Enum.GetValues(typeof(ActivationFunctionEnum)).Cast<ActivationFunctionEnum>()
				.First(f => f.ToString().Equals(activationFunction.Split('.').Last(), StringComparison.InvariantCultureIgnoreCase)); ;
		}
	}
}
