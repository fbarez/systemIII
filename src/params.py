from tkinter import W
from typing import Optional, Union

class Params:
    def __init__(self,
            state_size:int, 
            action_size:int, 
            hidden_size1:int=256, 
            hidden_size2:int=256,
            actions_continuous:bool=True,
            learning_rate:float=0.0003,
            reward_decay:float=0.99,
            gae_lambda:float=0.95,
            policy_clip:float=0.2,
            action_std_init:float=0.6,
            use_cuda:bool=True,
            checkpoint_dir:str="tmp/model",
            instance_name:str="",
            agent_type:str="ppo",
            game_mode:str="cartpole",
            num_timesteps:int=10000,
            num_iter:int=20,
            batch_size:int=5,
            n_epochs:int=4,
            test_period:int=500,
            test_iter:int=1000,
            timestep_length:Union[int,float]=1
            ):
        
        # initialize hyperparameters / config
        self.state_size   = state_size
        self.action_size  = action_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.actions_continuous = actions_continuous
        
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.action_std = action_std_init

        self.use_cuda = use_cuda

        self.checkpoint_dir = checkpoint_dir
        self.instance_name = instance_name
        self.agent_type = agent_type
        self.game_mode = game_mode

        self.num_timesteps = num_timesteps
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.test_period = test_period
        self.test_iter = test_iter
        self.timestep_length = timestep_length