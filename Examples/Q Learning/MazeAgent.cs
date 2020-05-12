using NeuralNetwork.Models;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeAgent : QAgent
{
    protected override int DoAction(int state, int action)
    {
        MazeEnv m = (MazeEnv)env;
        switch (action)
        {
            case 0:
                transform.position = m.State2Pos(state) + Vector3.forward;
                return m.Pos2State(transform.position);

            case 1:
                transform.position = m.State2Pos(state) + Vector3.back;
                return m.Pos2State(transform.position);

            case 2:
                transform.position = m.State2Pos(state) + Vector3.left;
                return m.Pos2State(transform.position);

            case 3:
                transform.position = m.State2Pos(state) + Vector3.right;
                return m.Pos2State(transform.position);

            default:
                throw new System.ArgumentOutOfRangeException("Action out of range.");
        }
    }
}
