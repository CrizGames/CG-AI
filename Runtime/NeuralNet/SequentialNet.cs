using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;
using System.IO;
using System.Threading.Tasks;

namespace CGAI.NeuralNetwork
{
    /// <summary>
    /// The base class for all neural networks
    /// </summary>
    [Serializable]
    public class SequentialNet
    {
        #region Fields

        private static JsonSerializerSettings settings = new JsonSerializerSettings { TypeNameHandling = TypeNameHandling.All };

        /// <summary>
        /// The layers of this neural network
        /// </summary>
        public Layer[] Layers;

        #endregion


        #region Constructors

        /// <summary>
        /// Create a neural network
        /// </summary>
        public SequentialNet(params Layer[] layers)
        {
            if (layers.Length < 2)
                throw new Exception($"Neural Network has {layers.Length} layer {(layers.Length > 1 || layers.Length == 0 ? "s" : "")}. There must be min. 2 layers.");

            Layers = layers;
        }

        #endregion


        #region Public Methods

        /// <summary>
        /// Initialize all layers of this neural network
        /// </summary>
        public virtual void Init(bool onlyPositiveWeights = false, float initWeightsRange = 5f)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layer layer = Layers[i];
                layer.Init(layer.Activations.Length, i > 0 ? Layers[i - 1] : null, layer.ActivationFunc, onlyPositiveWeights, initWeightsRange);
            }
        }

        /// <summary>
        /// Let the neural network run with specified input
        /// </summary>
        public virtual float[] FeedForward(List<float> input)
        {
            return FeedForward(input.ToArray());
        }

        /// <summary>
        /// Let the neural network run with specified input
        /// </summary>
        public virtual float[] FeedForward(float[] input)
        {
            // Check input
            CheckInput(input);

            // Set input values
            SetInputValues(input);

            // Go through each layer
            for (int i = 0; i < Layers.Length; i++)
                Layers[i].Process();

            // Return the output
            return GetOutput();
        }

        /// <summary>
        /// Returns the length of last layer
        /// </summary>
        public virtual int GetOutputLength()
        {
            return Layers.Last().Activations.Length;
        }

        /// <summary>
        /// Checks everything for input
        /// </summary>
        public virtual bool CheckInput(float[] input)
        {
            if (input == null)
                throw new NullReferenceException($"Input is null.");

            if (!CorrectInputLength(input.Length))
                throw new Exception($"The input has not the right length. Input length:{input.Length} Expected:{Layers[0].Activations.Length}");

            return true;
        }

        /// <summary>
        /// Check if the input length is equal to neurons in first layer.
        /// </summary>
        public virtual bool CorrectInputLength(int inputLength)
        {
            return Layers[0].Activations.Length == inputLength;
        }

        /// <summary>
        /// Save model to path
        /// </summary>
        public void Save(string path)
        {
            string json = JsonConvert.SerializeObject(Layers, Formatting.Indented, settings);

            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Save model to path
        /// </summary>
        public async Task SaveAsync(string path)
        {
            await Task.Run(() =>
            {
                string json = JsonConvert.SerializeObject(Layers, Formatting.Indented, settings);

                File.WriteAllText(path, json);
            });
        }

        /// <summary>
        /// Load model from path
        /// </summary>
        public bool Load(string path)
        {
            if (!File.Exists(path))
                return false;

            Layers = JsonConvert.DeserializeObject<Layer[]>(File.ReadAllText(path), settings);
            return true;
        }

        /// <summary>
        /// Load model from path
        /// </summary>
        public async Task<bool> LoadAsync(string path)
        {
            return await Task.Run(() =>
            {
                if (!File.Exists(path))
                    return false;

                Layers = JsonConvert.DeserializeObject<Layer[]>(File.ReadAllText(path), settings);
                return true;
            });
        }

        /// <summary>
        /// Prints weights and biases of neural net to the console
        /// </summary>
        public virtual void PrintModelInfo(bool detailed = false)
        {
            int weightsCount = 0;
            int biasesCount = 0;

            string modelStructureStr = "Model Structure:";
            List<string> layerStructureStrs = new List<string>();
            for (int i = 0; i < Layers.Length; i++)
            {
                Layer l = Layers[i];

                int layerWeightsCount = 0;
                int layerBiasesCount = 0;
                if (!l.IsInputLayer)
                {
                    // Get weights count
                    for (int j = 0; j < l.Weights.Length; j++)
                        layerWeightsCount += l.Weights[j].Length;
                    weightsCount += layerWeightsCount;

                    // Get biases count
                    layerBiasesCount = l.Biases.Length;
                    biasesCount += layerBiasesCount;
                }

                layerStructureStrs.Add(
                    $"Layer {i}:" +
                    $"\n    Neurons: {l.Activations.Length}" +
                    $"\n    Total Variables: {layerBiasesCount + weightsCount}" +
                    $"\n    Weights: {layerWeightsCount}" +
                    $"\n    Biases: {layerBiasesCount}\n\n");
            }

            modelStructureStr += $"\n    Total Weights: {weightsCount}" +
                $"\n    Total Biases: {biasesCount}\n\n";
            foreach (string str in layerStructureStrs)
                modelStructureStr += str;

            // TODO: Detailed mode

            Debug.Log(modelStructureStr);
        }

        #endregion


        #region Private Methods

        /// <summary>
        /// Returns the output
        /// </summary>
        /// <returns></returns>
        protected virtual float[] GetOutput()
        {
            return Layers.Last().Activations;
        }

        /// <summary>
        /// Sets the input
        /// </summary>
        protected virtual void SetInputValues(float[] input)
        {
            Layers[0].Activations = input;
        }

        #endregion
    }

}
