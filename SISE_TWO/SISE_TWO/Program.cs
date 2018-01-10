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

            double x;

            while (true)
            {
                Double.TryParse(Console.ReadLine(), out x);
                network.Compute(x);
            }



            Console.ReadKey();
        }

    }
}
