﻿using NeuralNetwork.ActivationFunctions.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork.ActivationFunctions
{
	internal class None : ActivationFunction
	{
		public override double F(double s)
		{
			return 1;
		}
	}
}