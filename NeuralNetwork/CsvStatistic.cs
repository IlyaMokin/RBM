using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
	public class CsvStatistic:IDisposable
	{

		private StreamWriter _writer;
		private bool _onConsole = true;
		private DateTime _startTime = DateTime.Now;

		public CsvStatistic(string filePath, bool onConsole = true, bool rewrite = false)
		{
			_writer = new StreamWriter(filePath, rewrite);
			_onConsole = onConsole;
			
			_writer.WriteLine("Statistic");
			
		}

		public CsvStatistic() { }

		public void Write(params object[] arguments)
		{
			var logString = string.Format("{0}; {1}",
					(DateTime.Now - _startTime).ToString(), arguments.Aggregate((x, y) => String.Format("{0};{1}", x, y)));

			_writer.WriteLine(logString);			

			if (_onConsole)
			{
				Console.WriteLine(logString);
			}
		}

		public void Dispose()
		{
			_writer.Close();
		}
	}
}
