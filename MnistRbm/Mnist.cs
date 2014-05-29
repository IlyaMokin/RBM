using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MnistRbm
{
	class Mnist
	{

		public IList<double[]> Read(string path)
		{
			var list = new List<double[]>();

			using (FileStream fStream = new FileStream(path, FileMode.Open))
			using (BinaryReader brImages = new BinaryReader(fStream))
			{
				int magic1 = brImages.ReadInt32(); // discard
				int numImages = brImages.ReadInt32();
				int numRows = brImages.ReadInt32();
				int numCols = brImages.ReadInt32();

				byte[] pixels = new byte[28 * 28];

				// each test image
				for (int di = 0; di < 10000; ++di)
				{
					for (int i = 0; i < 28 * 28; ++i)
					{
						byte b = brImages.ReadByte();
						pixels[i] = b;
					}
					list.Add(pixels.Select(x => (double)x).ToArray());
				}

			}
			return list;
		}
	}
}
