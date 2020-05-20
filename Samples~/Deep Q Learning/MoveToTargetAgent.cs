using CGAI.QLearning;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveToTargetAgent : DeepQAgent<MoveToTargetEnv>
{
    private void Start()
    {
        StartCoroutine(Train());
    }

    //Vector3 lastPos;
    //private void Update()
    //{
    //    if (lastPos == transform.position)
    //        return;

    //    float[] state = env.GetState(lastPos);

    //    print(env.RewardAgent(null, state));
    //    lastPos = transform.position;
    //}

    protected override float[] DoAction(float[] state, int action)
    {
        switch (action)
        {
            case 0:
                transform.position += Vector3.right;
                break;

            case 1:
                transform.position += Vector3.left;
                break;

            default:
                throw new System.ArgumentOutOfRangeException("Action out of range.");
        }
        return env.GetState(transform.position);
    }
}
