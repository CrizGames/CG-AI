using CGAI.QLearning;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeAgent : QAgent<MazeEnv>
{
    protected override int DoAction(int state, int action)
    {
        switch (action)
        {
            case 0:
                transform.position = env.State2Pos(state) + Vector3.forward;
                return env.Pos2State(transform.position);

            case 1:
                transform.position = env.State2Pos(state) + Vector3.back;
                return env.Pos2State(transform.position);

            case 2:
                transform.position = env.State2Pos(state) + Vector3.left;
                return env.Pos2State(transform.position);

            case 3:
                transform.position = env.State2Pos(state) + Vector3.right;
                return env.Pos2State(transform.position);

            default:
                throw new System.ArgumentOutOfRangeException("Action out of range.");
        }
    }
}
