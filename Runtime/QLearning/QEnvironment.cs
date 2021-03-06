﻿using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CGAI.QLearning
{
    public abstract class QEnvironment<T> : MonoBehaviour
    {
        public int StatesCount;

        /// <summary>
        /// Returns new state, reward and if goal reached 
        /// </summary>
        public virtual Tuple<float, bool> Step(T state, T newState)
        {
            return new Tuple<float, bool>(RewardAgent(state, newState), AchievedGoal(state, newState));
        }

        public abstract float RewardAgent(T state, T newState);

        public abstract bool AchievedGoal(T state, T newState);

        public abstract bool Failed(T state, T newState);

        public abstract T Reset();
    }
}