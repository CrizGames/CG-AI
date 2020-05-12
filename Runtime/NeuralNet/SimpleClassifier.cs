using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.NeuralNetwork
{
    public class SimpleClassifier : SequentialNet
    {
        /// <summary>
        /// This is a simple neural network with Leaky ReLU in the hidden layers and softmax in the output layer
        /// </summary>
        public SimpleClassifier(params int[] layers)
        {
            Layers = new Layer[layers.Length];

            // Set first layer without last layer
            Layers[0] = new Layers.Dense(layers[0], Activations.LeakyReLU);

            // Go through all layers except the last and create them with specified neurons and Leaky ReLU as thier activation functions
            for (int i = 1; i < layers.Length - 1; i++)
            {
                Layers[i] = new Layers.Dense(layers[i], Layers[i - 1], Activations.LeakyReLU);
            }

            // Set last layer with softmax
            int lastIdx = layers.Length - 1;
            Layers[lastIdx] = new Layers.Dense(layers[lastIdx], Layers[lastIdx - 1], Activations.Softmax);
        }
    }
}
