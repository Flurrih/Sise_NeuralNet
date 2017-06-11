using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SISE_TWO
{
    class NeuralNet
    {
        int numNeurons;
        int numLearnNeurons;
        int numInputData;
        int numOutputData;
        double learnRate;

        double[] inputData;
        double[] outputData;
        double[] expectedOutput;

        double[,] inputWeights;
        double[] hiddenWeights;

        public NeuralNet()
        {
            numNeurons = 100;
            numInputData = 5;
            numOutputData = 5;
            numLearnNeurons = 5;
            learnRate = 0.0001;
            inputData = new double[numInputData];
            expectedOutput = new double[numOutputData];
            inputWeights = new double[numLearnNeurons, numNeurons];
            hiddenWeights = new double[numNeurons];
            outputData = new double[numOutputData];

        }
        public void GenerateRandom()
        {
            Random rand = new Random();
            for (int i = 0; i < numInputData; i++)
            {
                inputData[i] = (double)rand.Next(1, 100) + (double)rand.Next(1, 10) / 10;
                expectedOutput[i] = Math.Sqrt(inputData[i]);
            }
            for (int i = 0; i < numNeurons; i++)
            {
                hiddenWeights[i] = (rand.NextDouble() - 0.5) / 10.0;
            }
            for (int i = 0; i < numLearnNeurons; i++)
            {
                for (int j = 0; j < numNeurons; j++)
                {
                    inputWeights[i, j] = (rand.NextDouble() - 0.5) / 10.0;
                }
            }

        }
        

        public void Run(int iter)
        {
            GenerateRandom();
            for (int i = 0; i < iter; i++)
            {
                double error = 0.0;
                for (int j = 0; j < numInputData; j++)
                {
                    double[] values = new double[numNeurons];
                    Train(j, values);
                    error += Math.Pow(outputData[j] - expectedOutput[j], 2);
                    AdjustWeights(j, values);
                }
                error = Math.Sqrt(error / numInputData);

                for (int j = 0; j < numInputData; j++)
                {
                    if (i == 0 || i == 100 || i % 2000 == 0)
                        Console.WriteLine("Epoka: " + i + " poczatkowa: " + inputData[j].ToString("0.0000") + " oczekiwana: " + expectedOutput[j].ToString("0.0000") + " otrzymana: " + outputData[j].ToString("0.0000") + " błąd względny " + Math.Abs(((expectedOutput[j] - outputData[j]) / outputData[j]) * 100).ToString("0.0000"));

                }
            }
        }

        public void Train(int inputIndex, double[] inputs)
        {
            for (int i = 0; i < numNeurons; i++)//neuron
            {
                inputs[i] = 0.0;
                for (int j = 0; j < numLearnNeurons; j++)//weight
                {
                    inputs[i] += inputData[inputIndex] * inputWeights[j, i];
                }
                inputs[i] = Sigmoid.Func(inputs[i]);
            }
            outputData[inputIndex] = 0.0;
            for (int i = 0; i < numNeurons; i++)
            {
                outputData[inputIndex] += hiddenWeights[i] * inputs[i];
            }
        }

        public void AdjustWeights(int inputIndex, double[] inputs)
        {
            for (int i = 0; i < numNeurons; i++)
            {
                hiddenWeights[i] -= (outputData[inputIndex] - expectedOutput[inputIndex]) * inputs[i] * learnRate;
                for (int j = 0; j < numLearnNeurons; j++)
                {
                    inputWeights[j, i] -= (outputData[inputIndex] - expectedOutput[inputIndex]) * Sigmoid.Derivative(inputs[i]) * hiddenWeights[i] * outputData[inputIndex] * learnRate;
                }
            }
        }
    }
}
