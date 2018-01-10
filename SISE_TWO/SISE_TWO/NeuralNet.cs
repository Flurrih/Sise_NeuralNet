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
            numInputData = 15;
            numOutputData = 15;
            numLearnNeurons = 15;
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
                //if(i == 0)
                //{
                //    inputData[i] = 1;
                //    expectedOutput[i] = Math.Sqrt(1);
                //}
                //if (i == 1)
                //{
                //    inputData[i] = 2;
                //    expectedOutput[i] = Math.Sqrt(2);
                //}
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
        double[] values;

        public void Run(int iter)
        {
            for (int i = 0; i <= iter; i++)
            {
                double error = 0.0;
                for (int j = 0; j < numInputData; j++)
                {
                    values = new double[numNeurons];
                    Train(j, values);
                    error += Math.Pow(outputData[j] - expectedOutput[j], 2);
                    AdjustWeights(j, values);
                }
                error = Math.Sqrt(error / numInputData);

                for (int j = 0; j < numInputData; j++)
                {
                    if (i == 0 || i == 100 || i % 2000 == 0)
                        Console.WriteLine("Epoch: " + i + " initial input: " + inputData[j].ToString("0.0000") + " expected: " + expectedOutput[j].ToString("0.0000") + " computed: " + outputData[j].ToString("0.0000") + " error " + Math.Abs(((expectedOutput[j] - outputData[j]) / outputData[j]) * 100).ToString("0.0000"));
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

        public void Compute(double x)
        {
            double outpt;
            for (int i = 0; i < numNeurons; i++)//neuron
            {
                values[i] = 0.0;
                for (int j = 0; j < numLearnNeurons; j++)//weight
                {
                    values[i] += x * inputWeights[j, i];
                }
                values[i] = Sigmoid.Func(values[i]);
            }
            outpt = 0.0;
            for (int i = 0; i < numNeurons; i++)
            {
                outpt += hiddenWeights[i] * values[i];
            }

            Console.WriteLine("input = " + x.ToString("0.0000") + " result = "+ outpt + " error = " + Math.Abs(((Math.Sqrt(x) - outpt) / outpt) * 100).ToString("0.0000"));
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

        public void Compute(double[] inputs, int inputIndex)
        {
            //double output=0;

            //for (int i = 0; i < numNeurons; i++)//neuron
            //{
            //    inputs[i] = 0.0;
            //    for (int j = 0; j < numLearnNeurons; j++)//weight
            //    {
            //        inputs[i] += inputData[inputIndex] * inputWeights[j, i];
            //    }
            //    inputs[i] = Sigmoid.Func(inputs[i]);
            //}
            //outputData[inputIndex] = 0.0;
            //for (int i = 0; i < numNeurons; i++)
            //{
            //    outputData[inputIndex] += hiddenWeights[i] * inputs[i];
            //}

            //Console.WriteLine(output.ToString());
        }

        public void Write()
        {
            Console.WriteLine(outputData[0]);
        }
    }
}
