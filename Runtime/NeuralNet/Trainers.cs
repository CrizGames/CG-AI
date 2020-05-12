using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace CGAI.NeuralNetwork
{
    public abstract class Trainer
    {
        public readonly SequentialNet NeuralNet;

        public Func<float[], float[], bool, float[]> ErrorFunc;

        public float LearningRate;

        private protected bool printError;

        /// <summary>
        /// Initialze the trainer object. The error function is used to decide how much the output is wrong and where.
        /// </summary>
        public Trainer(SequentialNet nn, Func<float[], float[], bool, float[]> errorFunction, float learningRate, bool printError = true)
        {
            NeuralNet = nn;

            ErrorFunc = errorFunction;
            if (ErrorFunc == null)
                throw new NullReferenceException("Error function is null.");

            LearningRate = learningRate;

            this.printError = printError;
        }

        /// <summary>
        /// Train the neural network in batches
        /// </summary>
        public abstract void TrainBatches(float[][] inputs, float[][] expectedOutputs, int batchSize, int epochs);

        /// <summary>
        /// Train the neural network in batches
        /// </summary>
        public abstract float TrainBatchesOnce(float[][] inputs, float[][] expectedOutputs, int batchSize = 4);

        /// <summary>
        /// Train the neural network once
        /// </summary>
        public abstract void Train(float[][] input, float[][] expectedOutput, int epochs);

        /// <summary>
        /// Train the neural network once
        /// </summary>
        public abstract float TrainOnce(float[] input, float[] expectedOutput);

        /// <summary>
        /// Train the neural network once
        /// </summary>
        protected abstract float TrainOnce(float[] errors);

        /// <summary>
        /// Lets the neural network run once and and calculates the error with the error function 
        /// </summary>
        public virtual float[] GetError(float[] input, float[] expectedOutput)
        {
            float[] output = NeuralNet.FeedForward(input);

            if (printError)
            {
                float[] e = ErrorFunc(output, expectedOutput, false);

                bool simple = false;
                if (simple)
                    Debug.Log($"Error: {Math.Round(e.Sum() / e.Length, 4)}");
                else
                {
                    string errorStr = $"Average error: {Math.Round(e.Sum() / e.Length, 4)}\n";
                    for (int i = 0; i < e.Length; i++)
                        errorStr += $"Out N{i}: {Math.Round(output[i], 2)}, Expected: {expectedOutput[i]}";

                    Debug.Log(errorStr);
                }
            }

            float[] error = ErrorFunc(output, expectedOutput, true);

            return error;
        }

        private protected void AdjustLayer(Layer layer, float[] dError)
        {
            // Adjust weights
            for (int n = 0; n < layer.Activations.Length; n++)
                for (int w = 0; w < layer.Weights[n].Length; w++)
                    layer.Weights[n][w] -= dError[n] * layer.LastLayer.Activations[w] * LearningRate;

            // Adjust weights
            for (int n = 0; n < layer.Activations.Length; n++)
                layer.Biases[n] -= dError[n] * LearningRate;
        }
    }

    public struct Trainers
    {
        public class BackPropagation : Trainer
        {
            public BackPropagation(SequentialNet nn, Func<float[], float[], bool, float[]> errorFunction, float learningRate, bool printError = true)
                : base(nn, errorFunction, learningRate, printError) { }


            /// <summary>
            /// Train the neural network in batches
            /// </summary>
            public override void TrainBatches(float[][] inputs, float[][] expectedOutputs, int batchSize, int epochs)
            {
                for (int i = 0; i < epochs; i++)
                    TrainBatchesOnce(inputs, expectedOutputs, batchSize);
            }

            /// <summary>
            /// Train the neural network in batches
            /// </summary>
            public override float TrainBatchesOnce(float[][] inputs, float[][] expectedOutputs, int batchSize = 4)
            {
                if (batchSize < 2)
                    throw new Exception("Batch size must be at least 2.");

                if (batchSize >= inputs.Length)
                    throw new Exception("Batch size is bigger than dataset length. Batch size must be at least equal the dataset length (Even though that's not the intended usage. It should be smaller.).");

                int startIdx = 0;
                int iterations = 0;
                float avgError = 0;

                // Get errors
                List<float[]> errors = new List<float[]>();
                for (int i = 0; i < inputs.Length; i++)
                    errors.Add(GetError(inputs[i], expectedOutputs[i]));

                do
                {
                    iterations++;

                    // Calculate average error
                    float[] averageErrors = new float[inputs[0].Length];
                    for (int n = 0; n < inputs[0].Length; n++)
                    {
                        float neuronErrorSum = 0;
                        for (int e = startIdx; e < startIdx + batchSize; e++)
                            neuronErrorSum += errors[e][n]; // TODO: Error here
                        averageErrors[n] = neuronErrorSum / errors.Count;
                    }

                    avgError += TrainOnce(averageErrors);

                    startIdx += batchSize;
                    if (startIdx > inputs.Length)
                        startIdx = inputs.Length - 1;
                }
                while (startIdx == inputs.Length - 1);

                return avgError / iterations;
            }

            /// <summary>
            /// Train the neural network once
            /// </summary>
            public override void Train(float[][] inputs, float[][] expectedOutputs, int epochs)
            {
                if (inputs.Length != expectedOutputs.Length)
                    throw new Exception("Inputs and Expected Outputs must be the same size");

                for (int i = 0; i < epochs; i++)
                    for (int j = 0; j < inputs.Length; j++)
                        TrainOnce(inputs[j], expectedOutputs[j]);
            }

            /// <summary>
            /// Train the neural network once
            /// </summary>
            public override float TrainOnce(float[] input, float[] expectedOutput)
            {
                return TrainOnce(GetError(input, expectedOutput));
            }

            /// <summary>
            /// Train the neural network once
            /// </summary>
            protected override float TrainOnce(float[] dError)
            {
                float avgError = dError.Sum() / dError.Length;

                Dictionary<Layer, float[]> errors = new Dictionary<Layer, float[]>();

                // Propagate backwards

                // Output layer
                Layer layer = NeuralNet.Layers.Last();

                // Get derivatives
                float[] derivatives = layer.ActivationFunc(layer.Activations, true);
                for (int n = 0; n < derivatives.Length; n++)
                    dError[n] *= derivatives[n];

                errors.Add(layer, dError);

                // All other layers
                for (int l = NeuralNet.Layers.Length - 2; l > 0; l--)
                {
                    layer = NeuralNet.Layers[l];
                    Layer nextLayer = NeuralNet.Layers[l + 1];

                    dError = new float[NeuralNet.Layers[l].Activations.Length];
                    float[] ds = layer.ActivationFunc(NeuralNet.Layers[l].Activations, true);
                    for (int n = 0; n < NeuralNet.Layers[l].Activations.Length; n++)
                    {
                        float layerSum = 0;
                        for (int i = 0; i < nextLayer.Activations.Length; i++)
                        {
                            layerSum += errors[nextLayer][i] * nextLayer.Weights[i][n];
                        }
                        dError[n] = layerSum * ds[n];
                    }
                    errors.Add(layer, dError);
                }

                foreach (var layerError in errors)
                    AdjustLayer(layerError.Key, layerError.Value);

                return avgError;
            }
        }
    }
}
