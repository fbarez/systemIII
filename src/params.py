from typing import Optional

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
            batch_size:Optional[int]=None,
            n_epochs:int=10,
            use_cuda:bool=True,
            checkpoint_dir:str="tmp/model",
            model_name:str="ppo"
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

        self.batch_size = batch_size
        self.n_epochs = n_epochs
 
        self.use_cuda = use_cuda

        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name