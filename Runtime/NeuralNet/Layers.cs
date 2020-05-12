using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.NeuralNetwork
{
    /// <summary>
    /// Base layer
    /// </summary>
    [Serializable]
    public abstract class Layer
    {
        /// <summary>
        /// A vector of all activations in this layer.
        /// </summary>
        public float[] Activations;

        /// <summary>
        /// A matrix of all weights in this layer.
        /// In each row are all weights for one neuron.
        /// </summary>
        public float[][] Weights;

        /// <summary>
        /// A vector of all biases in this layer.
        /// </summary>
        public float[] Biases;

        /// <summary>
        /// Here you can use the activation functions from NeuralNetwork.Activations or your own.
        /// If you want to add your own activation function, then the first parameter are the neurons and the second is if it should return the derivative or not. It must return a float.
        /// <example>
        /// Example: <c>public static float[] ReLU(float[] layer, bool derivative = false)</c>
        /// </example>
        /// </summary>
        public Func<float[], bool, float[]> ActivationFunc;

        /// <summary>
        /// The layer behind this layer
        /// </summary>
        public Layer LastLayer { get; private protected set; }

        /// <summary>
        /// Each element is one dimension with Length combined.
        /// </summary>
        public int[] InputShape { get; private protected set; }

        /// <summary>
        /// Determines if it is the first layer based on if "LastLayer" is null.
        /// </summary>
        public bool IsInputLayer { get; private protected set; }

        /// <summary>
        /// Initialize layer
        /// </summary>
        public abstract void Init(int neurons, Layer lastLayer, Func<float[], bool, float[]> activationFunc, bool onlyPositiveWeights, float initWeightsRange);

        public abstract void Process();
    }

    /// <summary>
    /// A struct containing all various layers
    /// </summary>
    public struct Layers
    {
        /// <summary>
        /// A layer with normal neurons
        /// </summary>
        [Serializable]
        public class Dense : Layer
        {
            /// <summary>
            /// Here specify how many neurons there should be in this layer. The default activation function is Leaky ReLU.
            /// If it is the first layer, simply put an empty layer as last layer.
            /// </summary>
            public Dense(int neurons)
            {
                // Create a dense vector with random values
                Activations = new float[neurons];

                ActivationFunc = CGAI.NeuralNetwork.Activations.LeakyReLU;
            }
            /// <summary>
            /// Here specify how many neurons there should be in this layer and which activation function it should use.
            /// If it is the first layer, simply put an empty layer as last layer.
            /// </summary>
            public Dense(int neurons, Func<float[], bool, float[]> activationFunc)
            {
                // Create a dense vector with random values
                Activations = new float[neurons];

                ActivationFunc = activationFunc;
            }

            /// <summary>
            /// Here specify how many neurons there should be in this layer and which activation function it should use.
            /// If it is the first layer, simply put an empty layer as last layer.
            /// </summary>
            public Dense(int neurons, Layer lastLayer, Func<float[], bool, float[]> activationFunc)
            {
                // Create a dense vector with random values
                Activations = new float[neurons];

                ActivationFunc = activationFunc;

                LastLayer = lastLayer;
            }

            /// <summary>
            /// Specify here how many neurons there should be in this layer. The activation function is Leaky ReLU by default.
            /// If it is the first layer, simply put an empty layer as last layer
            /// </summary>
            public Dense(int neurons, Layer lastLayer) : this(neurons, lastLayer, CGAI.NeuralNetwork.Activations.LeakyReLU) { }

            /// <summary>
            /// Initialize layer with these variables
            /// </summary>
            public override void Init(int neurons, Layer lastLayer, Func<float[], bool, float[]> activationFunc, bool onlyPositiveWeights, float initWeightsRange = 5f)
            {
                if (neurons <= 0)
                    throw new Exception("Neurons must be more than 0.");

                LastLayer = lastLayer;

                // Determine if this is the first layer
                IsInputLayer = lastLayer == null;

                if (Activations == null)
                    Activations = new float[neurons];

                // If first layer, set all unnecessary things to null
                if (IsInputLayer)
                {
                    Weights = null;
                    Biases = null;
                    ActivationFunc = CGAI.NeuralNetwork.Activations.Identity;
                }
                else
                {
                    System.Random rnd = new System.Random(DateTime.Now.Millisecond);

                    // Create a dense matrix with random values using he initialization
                    Weights = new float[neurons][];
                    for (int n = 0; n < Weights.Length; n++)
                    {
                        int ws = lastLayer.Activations.Length;
                        Weights[n] = new float[ws];

                        for (int w = 0; w < ws; w++)
                        {
                            float rndW = ((float)rnd.NextDouble() * 2f - 1f) * initWeightsRange;

                            if (onlyPositiveWeights)
                                rndW = (float)rnd.NextDouble() * initWeightsRange;

                            Weights[n][w] = rndW;
                        }
                    }

                    // Create a dense vector
                    Biases = new float[neurons];

                    // Specify which activation functions are used
                    if (ActivationFunc == null)
                        ActivationFunc = activationFunc;
                }
            }

            /// <summary>
            /// Multiply the weights by activations of last layer plus the biases.
            /// <example> 
            /// <code>
            /// activation function(weights * activations of last layer + biases)
            /// </code>
            /// </example>
            /// </summary>
            public override void Process()
            {
                if (IsInputLayer)
                    return;

                float[] neuronSums = new float[Activations.Length];
                for (int n = 0; n < Weights.Length; n++)
                {
                    for (int w = 0; w < Weights[n].Length; w++)
                        neuronSums[n] += LastLayer.Activations[w] * Weights[n][w];

                    neuronSums[n] += Biases[n];
                }

                Activations = ActivationFunc(neuronSums, false);
            }
        }
    }
}
