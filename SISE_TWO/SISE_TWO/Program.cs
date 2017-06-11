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
            network.Run(10000);
            Console.ReadKey();
        }
        
    }
}
