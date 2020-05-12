using CGAI.QLearning;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MazeEnv : QEnvironment
{
    public int mazeWidth = 20;
    public int mazeHeight = 20;

    public Transform obstaclesParent;
    private List<Vector3> obstacles;

    public Transform goal;
    private int goalState;

    public Transform agent;

    private void Awake()
    {
        obstacles = new List<Vector3>();
        for (int i = 0; i < obstaclesParent.childCount; i++)
            obstacles.Add(obstaclesParent.GetChild(i).position);

        goalState = Pos2State(goal.position);
    }

    public override bool AchievedGoal(int state, int newState)
    {
        return State2Pos(newState) == goal.position;
    }

    public override bool Failed(int state, int newState)
    {
        return obstacles.Contains(State2Pos(newState));
    }

    public override int Reset()
    {
        int rndState;
        Vector3 pos;
        do
        {
            rndState = Random.Range(0, StatesCount);
            pos = State2Pos(rndState);
        }
        while (goal.position == pos || obstacles.Contains(pos));

        return rndState;
    }

    public override float RewardAgent(int state, int newState)
    {
        if (agent.position.x < 0 || agent.position.y < 0 || agent.position.x >= mazeWidth || agent.position.y >= mazeHeight)
            return -10f;

        if (newState == goalState)
            return 10f;
        else if (Mathf.Abs(newState - goalState) < Mathf.Abs(state - goalState))
            return 0.1f;
        else
            return -0.1f;
    }

    public int Pos2State(Vector3 pos)
    {
        return Mathf.RoundToInt(pos.x + pos.z * mazeWidth);
    }

    public Vector3 State2Pos(int state)
    {
        return new Vector3(state % mazeWidth, 0, state / mazeHeight);
    }

    private void OnValidate()
    {
        StatesCount = mazeWidth * mazeHeight;
    }
}
