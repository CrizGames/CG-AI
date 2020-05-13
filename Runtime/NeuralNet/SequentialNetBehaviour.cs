using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Malee;

namespace CGAI.NeuralNetwork
{
    public class SequentialNetBehaviour : MonoBehaviour
    {
        [System.Serializable]
        private struct DenseLayer
        {
            public int neurons;
            public ActivationType func;

            public Layer ToLayer()
            {
                System.Func<float[], bool, float[]> activationFunc = null;
                switch (func)
                {
                    case ActivationType.Identity:
                        activationFunc = Activations.Identity;
                        break;

                    case ActivationType.BinaryStep:
                        activationFunc = Activations.BinaryStep;
                        break;

                    case ActivationType.ReLU:
                        activationFunc = Activations.ReLU;
                        break;

                    case ActivationType.LeakyReLU:
                        activationFunc = Activations.LeakyReLU;
                        break;

                    case ActivationType.Swish:
                        activationFunc = Activations.Swish;
                        break;

                    case ActivationType.Sigmoid:
                        activationFunc = Activations.Sigmoid;
                        break;

                    case ActivationType.Tanh:
                        activationFunc = Activations.Tanh;
                        break;

                    case ActivationType.Softmax:
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
        private DenseLayerArray layers = new DenseLayerArray();


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
                    layers.Add(new DenseLayer { neurons = 1, func = ActivationType.LeakyReLU });

            if (layers[0].func != ActivationType.Identity)
                layers[0] = new DenseLayer { neurons = layers[0].neurons, func = ActivationType.Identity };
        }
    }
}