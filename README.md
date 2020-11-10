# CG-AI
This is Criz Games' Artificial Intelligence Library for Unity.
 
It is not the fastest library, but fast enough for my purposes. It is probably better to use [Unity's ML-Agents](https://github.com/Unity-Technologies/ml-agents).
 
## Installation
In the Unity package manager, click on the plus sign > Add package from Git URL..  
1. `https://github.com/cfoulston/Unity-Reorderable-List.git` 
2. `jillejr.newtonsoft.json-for-unity`  Follow the steps for installing this library [here](https://github.com/jilleJr/Newtonsoft.Json-for-Unity/wiki/Installation-via-UPM).
3. `https://github.com/CrizGames/CG-AI.git`  

![screenshot](https://user-images.githubusercontent.com/38948134/81692046-1b0e9a80-945e-11ea-956f-607265b609df.png)  

## Neural Networks
When you want to make a simple neural network, then using `SequentialNet` is the right way to go.
```c#
SequentialNet net = new SequentialNet(
      new Layers.Dense(6, Activations.Identity),
      new Layers.Dense(8, Activations.ReLU),
      new Layers.Dense(8, Activations.Sigmoid),
      new Layers.Dense(4, Activations.Softmax));
net.Init();
```
As you can see, the library has a number of activation functions.

### Custom Layer
Currently, there is only the `Dense` layer, but you can add your own layer easily. Just inherit from the `Layer` class.
```c#
public class CustomLayer : Layer
{
    public override void Init(int neurons, Layer previousLayer, Func<float[], bool, float[]> activationFunc, bool onlyPositiveWeights, float initWeightsRange)
    {
        // Initialize layer
    }

    public override void Process()
    {
        // Feed forward
    }
}
```

### Custom Activation Function
Activation functions can also be added easily. Just create a function with `float[]` and `bool` as parameters and `float[]` as return type.
```c#
public static float[] CustomActivationFunction(float[] layer, bool derivative = false)
{
    if (derivative)
    {
        float[] derivatives = new float[layer.Length];

        // Do stuff if derivative

        return derivatives;
    }

    // Do stuff

    return layer;
}
```
