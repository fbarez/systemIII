"""Define Params class to hold all hyperparameters for the run, and provide defaults
"""
from typing import Union
import json

class Params:
    """Params class to hold all hyperparameters for the run, and provide defaults"""
    def __init__(self,
            state_size:int,
            action_size:int,
            hidden_size1:int = 256,
            hidden_size2:int = 256,
            actions_continuous:bool = True,
            use_cuda:bool = True,

            learning_rate:float = 0.0003,
            reward_decay:float = 0.99,
            gae_lambda:float = 0.95,
            clipped_advantage:bool = False,
            policy_clip:float = 0.2,
            action_std_init:float = 0.6,
            kl_target:float = 0,
            cumulative_limit:float = float("inf"),
            normalize_advantages:bool = True,
            normalization_epsilon:float = 1e-6,

            reward_penalized:bool = False,
            cost_decay:float = 0.99,
            cost_lambda:float = 0.95,
            train_cost_critic:bool = True,

            learn_penalty:bool = False,
            cost_limit:int = 25,
            penalty_init:float = 1.,
            penalty_lr:float = 5e-2,
            penalty_param_loss:bool = True,
            entropy_regularization:float = 0.00,

            checkpoint_dir:str = "tmp/model",
            instance_name:str = "",
            agent_type:str = "ppo",
            game_mode:str = "cartpole",
            num_timesteps:int = 10000,
            num_iter:int = 20,
            batch_size:int = 5,
            n_epochs:int = 4,

            run_tests:bool = True,
            test_period:int = 500,
            test_iter:int = 1000,
            timestep_length:Union[int,float] = 1,
            save_period:int = 0,

            dist_lower_bound:float = 0.3,
            dist_upper_bound:float = 0.6,
        ):

        # initialize hyperparameters / config
        self.state_size   = state_size
        self.action_size  = action_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.actions_continuous = actions_continuous
        self.use_cuda = use_cuda

        # Learn actor policy
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        # reward advantage parameters
        self.gae_lambda = gae_lambda
        self.clipped_advantage = clipped_advantage
        self.policy_clip = policy_clip
        self.action_std = action_std_init
        self.kl_target = kl_target
        self.cumulative_limit = cumulative_limit # TODO: reimplement
        self.normalize_advantages = normalize_advantages
        self.normalization_epsilon = normalization_epsilon

        # cost advantage parameters
        self.reward_penalized = reward_penalized
        self.cost_decay = cost_decay
        self.cost_lambda = cost_lambda

        # Learn value/cost critic
        # (no hyperparameters for value critic yet)
        self.train_cost_critic = train_cost_critic

        # Learn penalty (in conjuction with cost critic)
        self.learn_penalty = learn_penalty
        self.cost_limit = cost_limit
        self.penalty_init = penalty_init
        self.penalty_lr = penalty_lr
        self.penalty_param_loss = penalty_param_loss

        # optional entropy regularization term
        self.entropy_regularization = entropy_regularization

        self.checkpoint_dir = checkpoint_dir
        self.instance_name = instance_name
        self.agent_type = agent_type
        self.game_mode = game_mode

        self.num_timesteps = num_timesteps
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.run_tests = run_tests
        self.test_period = test_period
        self.test_iter = test_iter
        self.timestep_length = timestep_length
        self.save_period = save_period

        # calculate distance from car to object, and use to calculate cost
        self.dist_lower_bound = dist_lower_bound
        self.dist_upper_bound = dist_upper_bound


    def _json( self ):
        keys = dir( self )
        data = {}
        for key in keys:
            if key[0] == '_':
                continue
            data[key] = getattr( self, key )
        return data

    def _update( self, data ):
        for key, value in data.items():
            if key[0] == '_':
                continue
            setattr( self, key, value )
        return self

    def __str__( self ):
        return self._json().__str__()

    # pylint: disable=unspecified-encoding
    def _dump( self, filename ):
        with open( filename, 'w' ) as f:
            json.dump( self._json(), f )

    # pylint: disable=unspecified-encoding
    def _load( self, filename ):
        with open( filename, 'r' ) as f:
            data = json.load( f )
            self._update( data )
        return self
