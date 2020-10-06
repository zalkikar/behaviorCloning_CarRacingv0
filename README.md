# Behavior Cloning for OpenAI's CarRacing-v0
```
|-- drive_manually.py    (with flag --collect_data, stores manual driving data in './data/')
|-- model.py             (agent in './models/')
|-- train_agent.py       (train agent)
|-- test_agent.py        (agent performance, stores results in './results/') 
```
Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).

Neural Network has a validation MSE loss of 0.06 and mean episode rewards of 450 using a current image input grayscaled to (1,96,96) and preprocessed.

```
python test_agent.py
``` 
gives the following results:

![Alt Text](https://github.com/zalkikar/behaviorCloning_CarRacingv0/blob/main/agenttest.gif)
