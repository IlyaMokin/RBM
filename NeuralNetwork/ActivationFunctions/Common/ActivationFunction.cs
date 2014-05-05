using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions.Common
{
	internal abstract class ActivationFunction
	{
		protected double _h = 1e-10;

		public abstract double F(double s);

		/// <summary>
		/// if don't have implement then won't calculate second derivative 
		/// </summary>
		public virtual double DF(double s)
		{
			double x1 = F(s + _h);
			double x2 = F(s);
			double res = (x1 - x2) / _h;
			return res;
		}

		public virtual double DDF(double s)
		{
			double x1 = F(s + _h);
			double x2 = F(s);
			double res = (x1 - x2) / _h;
			return res;
		}
	}
}
