using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.QLearning
{
    public abstract class QEnvironment : MonoBehaviour
    {
        public int StatesCount;

        /// <summary>
        /// Returns new state, reward and if goal reached 
        /// </summary>
        public virtual Tuple<float, bool> Step(int state, int newState)
        {
            return new Tuple<float, bool>(RewardAgent(state, newState), AchievedGoal(state, newState));
        }

        public abstract float RewardAgent(int state, int newState);

        public abstract bool AchievedGoal(int state, int newState);

        public abstract bool Failed(int state, int newState);

        public abstract int Reset();
    }
}