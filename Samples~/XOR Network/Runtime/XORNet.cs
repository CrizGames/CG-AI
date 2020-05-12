using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using NeuralNetwork;

namespace NeuralNetwork.Models
{
    public class XORNet : MonoBehaviour
    {
        private readonly float[][][] dataTemplate =
            new float[][][] {
        new float[][] { new float[] { 0, 0 }, new float[] { 0 } },
        new float[][] { new float[] { 0, 1 }, new float[] { 1 } },
        new float[][] { new float[] { 1, 1 }, new float[] { 0 } },
        new float[][] { new float[] { 1, 0 }, new float[] { 1 } },
            };

        private SequentialNet nn;

        public SequentialNetBehaviour nnBehaviour;

        /// <summary>
        /// Start
        /// </summary>
        private void Start()
        {
            nn = nnBehaviour.GetSequentialNet();

            nn.Init(true, 1f);

            nn.PrintModelInfo(true);

            Trainer trainer = new Trainers.BackPropagation(nn, Errors.MeanSquaredError, 0.05f);

            float[][] inputs = new float[dataTemplate.Length][];
            float[][] expectedOutputs = new float[dataTemplate.Length][];

            for (int i = 0; i < inputs.Length; i++)
            {
                float[][] sample = dataTemplate[i];
                inputs[i] = sample[0];
                expectedOutputs[i] = sample[1];
            }

            trainer.Train(inputs, expectedOutputs, 400);
        }

        /// <summary>
        /// Update
        /// </summary>
        private void Update()
        {
            if (Input.GetKeyDown(KeyCode.Return))
            {
                float[][] sample = GenerateSample();
                float[] result = nn.FeedForward(sample[0]);

                // Print result
                Debug.Log(
                    $"Input: {System.Math.Round(sample[0][0], 1)} {System.Math.Round(sample[0][1], 2)}, " +
                    $"Output: {System.Math.Round(result[0], 2)}, " +
                    $"Expected Output: {System.Math.Round(sample[1][0], 2)}");
            }
        }

        private int j = 0;
        /// <summary>
        /// Get a training sample
        /// </summary>
        private float[][] GenerateSample()
        {
            j++;
            if (j > 3) j = 0;

            return dataTemplate[j];
        }
    }
}