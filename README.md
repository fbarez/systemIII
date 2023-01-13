# Safety Gym Constraints

## Prerequisites:

Must install OpenAI [safety-gym](https://github.com/openai/safety-gym) and dependencies.
In particular, [mujoco-200](https://www.roboti.us/download.html) is the most difficult.

The reccomended python version is 3.7, since these packages are old.

## Getting started

To install and train a PPO Cartpole model:
```
git clone https://github.com/fbarez/sgym-const
cd sgym-const/src
python script_train_model.py --game_mode cartpole --agent_type ac
```

to train a different model, you can use different arguments. For example to train the System3 agent on the OpenAI CarGoal game mode you can use the following:
```
python script_train_model.py --game_mode car --agent_type s3
```

To make more adjustments, the best thing to do is open `script_train_model.py` and modify `Params`
