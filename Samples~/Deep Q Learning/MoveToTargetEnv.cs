using CGAI.QLearning;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MoveToTargetEnv : QEnvironment<float[]>
{
    public Transform agent;
    public Transform goal;

    private Vector3 lastAgentPos;

    private void Awake()
    {
        lastAgentPos = agent.position;
    }

    public override bool AchievedGoal(float[] state, float[] newState)
    {
        return Mathf.RoundToInt(agent.position.x) == Mathf.RoundToInt(goal.position.x);
    }

    public override bool Failed(float[] state, float[] newState)
    {
        return transform.position.x > 30 || transform.position.x < 30;
    }

    public override float[] Reset()
    {
        agent.position = Vector3.zero;

        do
            goal.position = Vector3.right * Random.Range(-20, 20);
        while (goal.position == Vector3.zero);

        return GetState(Vector3.zero);
    }

    public override float RewardAgent(float[] state, float[] newState)
    {
        if (AchievedGoal(state, newState))
            return 1f;

        float distDelta = Vector3.Distance(lastAgentPos, goal.position) - Vector3.Distance(agent.position, goal.position);

        lastAgentPos = agent.position;

        if (distDelta >= 0)
            return 0.1f;
        else
            return -0.1f;
    }

    public float[] GetState(Vector3 pos)
    {
        float dir = goal.position.x - pos.x;

        if (dir >= 0)
            return new float[] { 0, 1 };
        else
            return new float[] { 1, 0 };
    }
}