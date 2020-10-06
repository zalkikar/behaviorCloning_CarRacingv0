# Behavior Cloning for OpenAI's CarRacing-v0

|-- drive_manually.py (with flag --collect_data, stores manual driving data in './data/')
|-- model.py          (agent in './models/')
|-- train_agent.py    (train agent)
|-- test_agent.py     (agent performance, stores results in './results/') 

Install pyTorch and gym requirements (I used an anaconda Python3.6 virtual env).
Agent had a final validation loss of 0.06 and mean episode rewards of 450.

Running test_agent.py gives the following results:

![Alt Text](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)