using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuralNetwork;
using Malee;

namespace NeuralNetwork.Models
{
    public class SequentialNetBehaviour : MonoBehaviour
    {
        [System.Serializable]
        private struct DenseLayer
        {
            public enum ActivationFunction
            {
                Identity,
                BinaryStep,
                ReLU,
                LeakyReLU,
                Swish,
                Sigmoid,
                Tanh,
                Softmax
            }

            public int neurons;
            public ActivationFunction func;

            public Layer ToLayer()
            {
                System.Func<float[], bool, float[]> activationFunc = null;
                switch (func)
                {
                    case ActivationFunction.Identity:
                        activationFunc = Activations.Identity;
                        break;

                    case ActivationFunction.BinaryStep:
                        activationFunc = Activations.BinaryStep;
                        break;

                    case ActivationFunction.ReLU:
                        activationFunc = Activations.ReLU;
                        break;

                    case ActivationFunction.LeakyReLU:
                        activationFunc = Activations.LeakyReLU;
                        break;

                    case ActivationFunction.Swish:
                        activationFunc = Activations.Swish;
                        break;

                    case ActivationFunction.Sigmoid:
                        activationFunc = Activations.Sigmoid;
                        break;

                    case ActivationFunction.Tanh:
                        activationFunc = Activations.Tanh;
                        break;

                    case ActivationFunction.Softmax:
                        activationFunc = Activations.Softmax;
                        break;
                }
                return new Layers.Dense(neurons, activationFunc);
            }
        }

        [System.Serializable]
        private class DenseLayerArray : ReorderableArray<DenseLayer> { }


        // Variables

        [SerializeField, Reorderable]
        private DenseLayerArray layers;


        // Methods

        public SequentialNet GetSequentialNet()
        {
            Layer[] ls = new Layer[layers.Count];
            for (int i = 0; i < layers.Count; i++)
                ls[i] = layers[i].ToLayer();

            return new SequentialNet(ls);
        }


        private void OnValidate()
        {
            if (layers.Count < 2)
                for (int i = 0; i <= 2 - layers.Count; i++)
                    layers.Add(new DenseLayer { neurons = 1, func = DenseLayer.ActivationFunction.LeakyReLU });

            if (layers[0].func != DenseLayer.ActivationFunction.Identity)
                layers[0] = new DenseLayer { neurons = layers[0].neurons, func = DenseLayer.ActivationFunction.Identity };
        }
    }
}