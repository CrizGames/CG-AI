using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

namespace CGAI.QLearning
{
    public abstract class QAgent<T> : MonoBehaviour where T : QEnvironment<int>
    {
        public T env;

        public float[][] QTable;

        public int ActionsSize;

        private float explorationRate = 1f;
        [Range(0.01f, 1f)] public float MinExplorationRate = 0.01f;
        [Range(0.01f, 1f)] public float MaxExplorationRate = 1f;
        public float ExplorationDecayRate = 0.001f;

        public float LearningRate = 0.5f;
        public float DiscountFactor = 0.99f;

        public int Episodes = 10000;
        public int MaxStepsPerEpisode = 100;

        private int currentState;

        private void Awake()
        {
            if (env == null)
                throw new Exception("Environment is null!");

            if (ActionsSize < 2)
                throw new Exception("Agent must have 2 or more actions");

            QTable = new float[env.StatesCount][];
            for (int i = 0; i < QTable.Length; i++)
                QTable[i] = new float[ActionsSize];

            explorationRate = MaxExplorationRate;
        }

        public void RunOnce()
        {
            int newState = DoAction(currentState, GetBestActionIdx(currentState));
            Tuple<float, bool> envData = env.Step(currentState, newState);
            bool goalAchieved = envData.Item2;

            if (goalAchieved || OutOfRange(newState) || env.Failed(currentState, newState))
                currentState = env.Reset();
            else
                currentState = newState;

            Invoke("RunOnce", 0.1f);
        }

        public IEnumerator Train()
        {
            DateTime startTime = DateTime.Now;
            float totalReward = 0f;
            int wins = 0, fails = 0;
            for (int episode = 0; episode < Episodes; episode++)
            {
                int state = env.Reset();

                float rewardsCurrentEpisode = 0;

                bool goalAchieved = false;
                for (int step = 0; step < MaxStepsPerEpisode; step++)
                {
                    int action;
                    // Exploit
                    if (UnityEngine.Random.value > explorationRate)
                        action = GetBestActionIdx(state);
                    // Explore
                    else
                        action = UnityEngine.Random.Range(0, ActionsSize);

                    // Step
                    int newState = DoAction(state, action);
                    Tuple<float, bool> envData = env.Step(state, newState);
                    float reward = envData.Item1;
                    goalAchieved = envData.Item2;

                    // Update Q-Table for Q(s,a)
                    bool outOfRange = OutOfRange(newState);
                    if (outOfRange)
                        QTable[state][action] += LearningRate * (reward - QTable[state][action]);
                    else
                        QTable[state][action] += LearningRate * (reward + DiscountFactor * GetBestAction(newState) - QTable[state][action]);

                    currentState = state = newState;
                    rewardsCurrentEpisode += reward;

                    // If goal reached
                    if (goalAchieved || outOfRange || env.Failed(state, newState))
                        break;
                }

                // Lower exploration rate
                explorationRate = MinExplorationRate + (MaxExplorationRate - MinExplorationRate) * Mathf.Exp(-ExplorationDecayRate * episode);

                totalReward += rewardsCurrentEpisode;

                if (goalAchieved)
                    wins++;
                else
                    fails++;

                if (episode % 1000 == 0)
                {
                    Debug.Log($"Episode {episode} Exploration rate: {explorationRate} Wins/Fails: {(fails > 0 ? Math.Round((float)wins / fails, 2) : 0)}");
                    wins = fails = 0;
                }
                if (MaxStepsPerEpisode < 5000)
                {
                    if (episode % 4000 == 0)
                        yield return null;
                }
                else if (episode % 1000 == 0)
                    yield return null;
            }

            Debug.Log($"Training finished!  Time: {Math.Round((DateTime.Now - startTime).TotalSeconds, 1)}s  Total reward: {totalReward}");
            PrintQTable();

            currentState = env.Reset();
            Invoke("RunOnce", 0.1f);
        }

        public virtual bool OutOfRange(int newState)
        {
            return newState < 0 || newState >= env.StatesCount;
        }

        protected abstract int DoAction(int state, int action);

        protected virtual float GetBestAction(int state)
        {
            return Mathf.Max(QTable[state]);
        }

        protected virtual int GetBestActionIdx(int state)
        {
            return Array.IndexOf(QTable[state], Mathf.Max(QTable[state]));
        }

        public virtual void PrintQTable()
        {
            string str = "Q Table:\n";
            for (int s = 0; s < env.StatesCount; s++)
            {
                str += $"State {s,4}:\t";
                for (int a = 0; a < ActionsSize; a++)
                {
                    str += $"{QTable[s][a],3} ";
                }
                str += "\n";
            }
            Debug.Log(str);
        }

#if UNITY_EDITOR
        private void OnValidate()
        {
            if (MaxExplorationRate < MinExplorationRate)
                MinExplorationRate = MaxExplorationRate;
        }
#endif
    }
}