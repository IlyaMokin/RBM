using NeuralNetwork.ActivationFunctions;
using NeuralNetwork.ActivationFunctions.Common;
using NeuralNetwork.NetworkComponents;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.NetworkComponents
{


	class Neuron
	{
		public List<Connection> InputConnections = new List<Connection>();
		public List<Connection> OutputConnections = new List<Connection>();

		public FunctionActivator ActivationF;
		public IList<FunctionActivator> Deriviations = new List<FunctionActivator>();

		public Stimulus NeuronStimulus = delegate { };

		public IList<double> Parameters = new List<double>();
		public double S;
		public double Out;
		public decimal Error;
	}
}
