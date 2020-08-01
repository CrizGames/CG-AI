using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using CGAI.NeuralNetwork;
using System.Linq;
using System;

namespace CGAI.QLearning
{
    [RequireComponent(typeof(SequentialNetBehaviour))]
    public abstract class DeepQAgent<T> : MonoBehaviour where T : QEnvironment<float[]>
    {
        public class ReplayMemory
        {
            public int Capacity { get; set; }

            private int SampleSize { get; set; }
            public bool CanGetSample { get { return SampleSize <= memory.Count; } }

            // State, Action, Reward, New State, Done
            private List<Tuple<float[], int, float, float[], bool>> memory = new List<Tuple<float[], int, float, float[], bool>>();

            private int pushCount;

            public ReplayMemory(int capacity, int sampleSize)
            {
                if (capacity < 2)
                    throw new System.Exception("Capacity must be at least 2");

                Capacity = capacity;
                SampleSize = sampleSize;
            }

            /// <summary>
            /// State, Action, Reward, New State, Done
            /// </summary>
            public void Push(Tuple<float[], int, float, float[], bool> experience)
            {
                if (memory.Count < Capacity)
                    memory.Add(experience);
                else
                    memory[pushCount % Capacity] = experience;
                pushCount++;
            }

            public void Push(float[] state, int action, float reward, float[] newState, bool done)
            {
                var experience = new Tuple<float[], int, float, float[], bool>(state, action, reward, newState, done);
                if (memory.Count < Capacity)
                    memory.Add(experience);
                else
                    memory[pushCount % Capacity] = experience;
                pushCount++;
            }

            public List<Tuple<float[], int, float, float[], bool>> GetSample()
            {
                if (!CanGetSample)
                    throw new Exception("Batch size must be smaller or equal than memory count.");

                var possibleMemories = memory.ToList(); // ToList == Copy
                var sample = new List<Tuple<float[], int, float, float[], bool>>();
                while (sample.Count < SampleSize)
                    sample.Add(possibleMemories[UnityEngine.Random.Range(0, possibleMemories.Count)]);

                return sample;
            }
        }

        public T env;

        public int replayMemoryCapacity = 100;
        public int memorySampleSize = 64;
        private ReplayMemory replayMemory;

        private SequentialNet policyNet;
        private Trainer policyNetTrainer;
        private SequentialNet targetNet;

        public int updateTargetNetEvery = 6;
        private int targetUpateCounter = 0;

        [Tooltip("Network output length must be equal to actions size")]
        public int ActionsSize;

        private float explorationRate = 1f;
        [Range(0.01f, 1f)] public float MinExplorationRate = 0.01f;
        [Range(0.01f, 1f)] public float MaxExplorationRate = 1f;
        public float ExplorationDecayRate = 0.001f;

        public float LearningRate = 0.05f;
        public float DiscountFactor = 0.99f;

        public int Episodes = 10000;
        public int MaxStepsPerEpisode = 100;

        private void Awake()
        {
            if (env == null)
                throw new Exception("Environment is null!");

            if (ActionsSize < 2)
                throw new Exception("Agent must have 2 or more actions");

            policyNet = GetComponent<SequentialNetBehaviour>().GetSequentialNet();
            policyNet.Init();
            UpdateTargetNet();

            policyNetTrainer = new Trainers.BackPropagation(policyNet, Errors.MeanSquaredError, LearningRate, false);


            replayMemory = new ReplayMemory(replayMemoryCapacity, memorySampleSize);

            explorationRate = MaxExplorationRate;
        }

        public IEnumerator Train()
        {
            int wins = 1, fails = 1;
            float episodeReward = 0;
            for (int episode = 0; episode < Episodes; episode++)
            {
                float[] currentState = env.Reset();

                bool done = false;
                int step = 0;
                while (!done && step < MaxStepsPerEpisode)
                {
                    int action;
                    // Exploit
                    if (UnityEngine.Random.value > explorationRate)
                        action = PredictBestActionIdx(currentState);
                    // Explore
                    else
                        action = UnityEngine.Random.Range(0, ActionsSize);

                    // Step
                    float[] newState = DoAction(currentState, action);
                    Tuple<float, bool> envData = env.Step(currentState, newState);
                    episodeReward += envData.Item1;
                    done = envData.Item2;

                    // Update memory and train
                    replayMemory.Push(currentState, action, envData.Item1, newState, done);
                    Learn(done);

                    currentState = newState;
                    step++;
                }

                UpdateExplorationRate(episode);

                if (done)
                    wins++;
                else
                    fails++;

                if (episode % 100 == 0)
                {
                    Debug.Log($"Episode {episode}  Exploration rate: {explorationRate}  Wins/Fails: {Math.Round((float)wins / fails, 2)}  Mean episode reward: {episodeReward / 100f}");
                    wins = fails = 1;
                    episodeReward = 0;
                }
                yield return null;
            }

            Invoke("RunOnce", 0.1f);
        }

        public void Learn(bool terminalState)
        {
            if (!replayMemory.CanGetSample)
                return;

            // State, Action, Reward, New State, Done
            List<Tuple<float[], int, float, float[], bool>> sample = replayMemory.GetSample();

            List<float[]> trainingInput = new List<float[]>();
            List<float[]> trainingOutput = new List<float[]>();

            List<float[]> currentQsList = new List<float[]>();
            List<float[]> futureQsList = new List<float[]>();
            for (int i = 0; i < sample.Count; i++)
            {
                // Split sample
                var currentSample = sample[i];
                float[] state = currentSample.Item1;
                int action = currentSample.Item2;
                float reward = currentSample.Item3;
                float[] newState = currentSample.Item4;
                bool done = currentSample.Item5;

                // Predict Q values from current states
                currentQsList.Add(policyNet.FeedForward(state));
                // Predict Q values from future states
                futureQsList.Add(targetNet.FeedForward(newState));

                // If not a terminal state, get new q from future states, otherwise set it to 0
                // almost like with Q Learning, but we use just part of equation here
                float newQ;
                if (!done)
                {
                    float maxFutureQ = GetBestAction(futureQsList.Last());
                    newQ = reward + DiscountFactor * maxFutureQ;
                }
                else
                    newQ = reward;

                float[] currentQs = currentQsList[i];
                currentQs[action] = newQ;

                trainingInput.Add(state);
                trainingOutput.Add(currentQs);
            }

            policyNetTrainer.TrainBatchesOnce(trainingInput.ToArray(), trainingOutput.ToArray());

            if (terminalState)
                targetUpateCounter++;

            if (targetUpateCounter >= updateTargetNetEvery)
            {
                UpdateTargetNet();
                targetUpateCounter = 0;
            }
        }

        private void UpdateTargetNet()
        {
            targetNet = new SequentialNet(policyNet.Layers);
            Layer[] layers = policyNet.Layers.ToArray();
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i] = layers[i].Copy();
                layers[i].IsInputLayer = i == 0;
                layers[i].LastLayer = i > 0 ? layers[i - 1] : null;
            }
            targetNet.Layers = layers;
        }

        protected abstract float[] DoAction(float[] state, int action);

        protected virtual float PredictBestAction(float[] state)
        {
            return GetBestAction(GetQs(state));
        }

        protected virtual float[] GetQs(float[] state)
        {
            return policyNet.FeedForward(state);
        }

        protected virtual int PredictBestActionIdx(float[] state)
        {
            return GetBestActionIdx(GetQs(state));
        }

        protected virtual int GetBestActionIdx(float[] actions)
        {
            return Array.IndexOf(actions, Mathf.Max(actions));
        }

        protected virtual float GetBestAction(float[] actions)
        {
            return Mathf.Max(actions);
        }

        public void UpdateExplorationRate(int episode)
        {
            explorationRate = MinExplorationRate + (MaxExplorationRate - MinExplorationRate) * Mathf.Exp(-ExplorationDecayRate * episode);
        }
    }
}