using SISE_TWO;
using System;
namespace CodingBackProp
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuralNet network = new NeuralNet();
            network.GenerateRandom();
            network.Run(50000);

            int n;

            Console.WriteLine("Podaj ile liczb wylosowac:");
            Int32.TryParse(Console.ReadLine(), out n);

            double[] tab = GenereteInput(n);

            for(int i = 0; i < n; i++)
            {
                network.Compute(tab[i]);
            }

            Console.ReadKey();
        }

        static double[] GenereteInput(int inputNumber)
        {
            Random rand = new Random();
            double[] tab = new double[inputNumber];

            for (int i = 0; i < inputNumber; i++)
            {
                tab[i] = (double)rand.Next(1, 100) + (double)rand.Next(1, 10) / 10;
            }
            return tab;
        }

    }
}
