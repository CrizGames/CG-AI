using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.NeuralNetwork
{
    /// <summary>
    /// This class includes all activation functions
    /// </summary>
    public struct Activations
    {
        /// <summary>
        /// The identity activation function does nothing. It simply return the input.
        /// </summary>
        public static float[] Identity(float[] layer, bool derivative = false)
        {
            if (derivative)
            {
                float[] derivatives = new float[layer.Length];
                for (int d = 0; d < derivatives.Length; d++)
                    derivatives[d] = 1f;
                return derivatives;
            }

            return layer;
        }

        /// <summary>
        /// The identity activation function does nothing. It simply return the input.
        /// Note: Backpropagation will not work with this activation function because the derivative is 0.
        /// </summary>
        public static float[] BinaryStep(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
                return neurons;

            for (int i = 0; i < layer.Length; i++)
                neurons[i] = layer[i] < 0f ? 0f : 1f;
            return neurons;
        }

        /// <summary>
        /// According to the paper from researchers at Google, Swish is often better as ReLU. 
        /// But for small networks, it is better to stay with ReLU because Swish won't make any significant diffrences and ReLU is faster in computing.
        /// Note: for the derivative, the values must be computed before.
        /// </summary>
        public static float[] ReLU(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                    neurons[i] = layer[i] > 0f ? 1f : 0f;
                return neurons;
            }

            for (int i = 0; i < layer.Length; i++)
                neurons[i] = Math.Max(0f, layer[i]);
            return neurons;
        }

        /// <summary>
        /// According to the paper from researchers at Google, Swish is often better as ReLU. 
        /// But for small networks, it is better to stay with ReLU because Swish won't make any significant diffrences and ReLU is faster in computing.
        /// Note: for the derivative, the values must be computed before.
        /// </summary>
        public static float[] LeakyReLU(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                    neurons[i] = layer[i] > 0f ? 1f : 0.05f;
                return neurons;
            }

            for (int i = 0; i < layer.Length; i++)
            {
                float n = layer[i];
                neurons[i] = Math.Max(0.05f * n, n);
            }
            return neurons;
        }

        /// <summary>
        /// Swish is a new, self-gated activation function discovered by researchers at Google. 
        /// Swish is simply x * Sigmoid(x). It is very similar to ReLU.
        /// According to their paper, it performs better than ReLU.
        /// In experiments on ImageNet with identical models running ReLU and Swish, Swish achieved top -1 classification accuracy 0.6-0.9% higher.
        /// The downside is that Swish is slower than ReLU.
        /// </summary>
        public static float[] Swish(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                {
                    float n = layer[i];
                    float s = n / (1.0f + (float)Math.Exp(-n));
                    neurons[i] = n + n * (1f - n);
                }
                return neurons;
            }

            for (int i = 0; i < layer.Length; i++)
            {
                float n = layer[i];
                neurons[i] = n / (1.0f + (float)Math.Exp(-n));
            }
            return neurons;
        }

        /// <summary>
        /// The Sigmoid Activation Function. Note: for the derivative, the values must be computed before
        /// </summary>
        public static float[] Sigmoid(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                {
                    float n = layer[i];
                    neurons[i] = n * (1f - n);
                }
                return neurons;
            }

            for (int i = 0; i < layer.Length; i++)
                neurons[i] = 1.0f / (1.0f + (float)Math.Exp(-layer[i]));
            return neurons;
        }

        /// <summary>
        /// The Tanh Activation Function. Note: for the derivative, the values must be computed before
        /// </summary>
        public static float[] Tanh(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                    neurons[i] = (1 - (float)Math.Pow(layer[i], 2));
                return neurons;
            }

            for (int i = 0; i < layer.Length; i++)
                neurons[i] = (float)Math.Tanh(layer[i]);
            return neurons;
        }

        /// <summary>
        /// The Softmax Activation Function. Note: for the derivative, the values must be computed before
        /// </summary>
        public static float[] Softmax(float[] layer, bool derivative = false)
        {
            float[] neurons = new float[layer.Length];

            if (derivative)
            {
                for (int i = 0; i < layer.Length; i++)
                {
                    float n = layer[i];
                    neurons[i] = (1f - n) * n;
                }
                return neurons;
            }

            float sum = 0;
            for (int i = 0; i < layer.Length; i++)
            {
                neurons[i] = (float)Math.Exp(layer[i]);
                sum += neurons[i];
            }
            for (int i = 0; i < layer.Length; i++)
                neurons[i] /= sum;
            return neurons;
        }
    }
}
