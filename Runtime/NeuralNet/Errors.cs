using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.NeuralNetwork
{
    public struct ErrorTools
    {
        /// <summary>
        /// Checks if the arrays will work together
        /// </summary>
        public static bool CheckOutputs(float[] output, float[] expectedOutput)
        {
            if (output == null)
                throw new NullReferenceException("Output is null.");

            if (expectedOutput == null)
                throw new NullReferenceException("Expected output is null.");

            if (output.Length != expectedOutput.Length)
                throw new Exception("Output and expected output are not the same size.");

            return true;
        }
    }

    public struct Errors
    {
        public static float[] MeanSquaredError(float[] output, float[] expectedOutput, bool derivative = true)
        {
            ErrorTools.CheckOutputs(output, expectedOutput);

            float[] neurons = new float[output.Length];
            if (derivative)
            {
                for (int i = 0; i < neurons.Length; i++)
                    neurons[i] = 2 * (output[i] - expectedOutput[i]);
                return neurons;
            }

            for (int i = 0; i < neurons.Length; i++)
                neurons[i] = (float)Math.Pow(output[i] - expectedOutput[i], 2d);
            return neurons;
        }
    }
}
