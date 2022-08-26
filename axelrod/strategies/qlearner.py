from collections import OrderedDict
from typing import Dict, Union

from axelrod.action import Action, actions_to_str
from axelrod.player import Player
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from collections import deque

Score = Union[int, float]

C, D = Action.C, Action.D


class RiskyQLearner(Player):
    """A player who learns the best strategies through the q-learning
    algorithm.

    This Q learner is quick to come to conclusions and doesn't care about the
    future.

    Names:

    - Risky Q Learner: Original name by Geraint Palmer
    """

    name = "Risky QLearner"
    classifier = {
        "memory_depth": float("inf"),  # Long memory
        "stochastic": True,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }
    learning_rate = 0.9
    discount_rate = 0.9
    action_selection_parameter = 0.1
    memory_length = 12

    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True

        self.prev_action = None  # type: Action
        self.original_prev_action = None  # type: Action
        self.score = 0
        self.Qs = OrderedDict({"": OrderedDict(zip([C, D], [0, 0]))})
        self.Vs = OrderedDict({"": 0})
        self.prev_state = ""

    def receive_match_attributes(self):
        (R, P, S, T) = self.match_attributes["game"].RPST()
        self.payoff_matrix = {C: {C: R, D: S}, D: {C: T, D: P}}

    def strategy(self, opponent: Player) -> Action:
        """Runs a qlearn algorithm while the tournament is running."""
        if len(self.history) == 0:
            self.prev_action = self._random.random_choice()
            self.original_prev_action = self.prev_action
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        if state not in self.Qs:
            self.Qs[state] = OrderedDict(zip([C, D], [0, 0]))
            self.Vs[state] = 0
        self.perform_q_learning(
            self.prev_state, state, self.prev_action, reward
        )
        action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
        return action

    def select_action(self, state: str) -> Action:
        """
        Selects the action based on the epsilon-soft policy
        """
        rnd_num = self._random.random()
        p = 1.0 - self.action_selection_parameter
        if rnd_num < p:
            return max(self.Qs[state], key=lambda x: self.Qs[state][x])
        return self._random.random_choice()

    def find_state(self, opponent: Player) -> str:
        """
        Finds the my_state (the opponents last n moves +
        its previous proportion of playing C) as a hashable state
        """
        prob = "{:.1f}".format(opponent.cooperations)
        action_str = actions_to_str(opponent.history[-self.memory_length :])
        return action_str + prob

    def perform_q_learning(
        self, prev_state: str, state: str, action: Action, reward
    ):
        """
        Performs the qlearning algorithm
        """
        self.Qs[prev_state][action] = (1.0 - self.learning_rate) * self.Qs[
            prev_state
        ][action] + self.learning_rate * (
            reward + self.discount_rate * self.Vs[state]
        )
        self.Vs[prev_state] = max(self.Qs[prev_state].values())

    def find_reward( self, opponent: Player ) -> Dict[Action, Dict[Action, Score]]:
        """
        Finds the reward gained on the last iteration
        """

        if len(opponent.history) == 0:
            opp_prev_action = self._random.random_choice()
        else:
            opp_prev_action = opponent.history[-1]
        return self.payoff_matrix[self.prev_action][opp_prev_action]


class ArrogantQLearner(RiskyQLearner):
    """A player who learns the best strategies through the q-learning
    algorithm.

    This Q learner jumps to quick conclusions and cares about the future.

    Names:

    - Arrogant Q Learner: Original name by Geraint Palmer
    """

    name = "Arrogant QLearner"
    learning_rate = 0.9
    discount_rate = 0.1


class HesitantQLearner(RiskyQLearner):
    """A player who learns the best strategies through the q-learning algorithm.

    This Q learner is slower to come to conclusions and does not look ahead much.

    Names:

    - Hesitant Q Learner: Original name by Geraint Palmer
    """

    name = "Hesitant QLearner"
    learning_rate = 0.1
    discount_rate = 0.9


class CautiousQLearner(RiskyQLearner):
    """A player who learns the best strategies through the q-learning algorithm.

    This Q learner is slower to come to conclusions and wants to look ahead
    more.

    Names:

    - Cautious Q Learner: Original name by Geraint Palmer
    """

    name = "Cautious QLearner"
    learning_rate = 0.1
    discount_rate = 0.1
    
class MediocreQLearner(RiskyQLearner):
    """A player who learns mediocre strategies through the q-learning algorithm.

    This Q learner is mediocre to come to conclusions and only looks mid way ahead

    Names:

    - Mediocre Q Learner: Original name by Davor Jovanoski
    """

    name = "Mediocre QLearner"
    learning_rate = 0.5
    discount_rate = 0.5
    
class DeepQLearner(RiskyQLearner):
    """A player who learns deep q-learning algorithm.


    Names:

    - DeepQ Learner: Original name by Davor Jovanoski
    """

    name = "Deep QLearner"   
    learning_rate = 0.7
    discount_rate = 0.3
    
    def __init__(self) -> None:
        """Initialises the player by picking a random strategy."""

        super().__init__()

        # Set this explicitly, since the constructor of super will not pick it up
        # for any subclasses that do not override methods using random calls.
        self.classifier["stochastic"] = True
        self.target_model = self.agent()
        self.model = self.agent()
        self.past = deque(maxlen=50)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.batch_size = 4
        self.copy_to_target_model()

    def agent(self):
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(16, input_shape=(1,1), activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(8, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def copy_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def note(self, state, action, reward, next_state, done):
        self.past.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self._random.random_choice()
        future_action = self.model.predict(state)
        return np.argmax(future_action[0])  # returns action
            
    def replay(self):
        minibatch = random.sample(self.past, self.batch_size)
        for state, action, reward, next_state in minibatch:
            target = self.model.predict(state)
            t = self.target_model.predict(next_state)[0]
            target[0][action] = reward + self.discount_rate * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
  
        
    def strategy(self, opponent: Player) -> Action:
        """Runs a qlearn algorithm while the tournament is running."""
        if len(self.history) == 0:
            self.prev_action = self._random.random_choice()
            self.original_prev_action = self.prev_action
        state = self.find_state(opponent)
        reward = self.find_reward(opponent)
        action = self.act(state) 

        if state not in self.Qs:
            self.Vs[state] = 0
            
        self.note(self.prev_state, self.prev_action, reward, state)  
              
        
        action = self.select_action(state)
        self.prev_state = state
        self.prev_action = action
 
        if len(self.past) > self.batch_size:
            self.replay()
        else:
            self.copy_to_target_model()
            
        self.prev_state = state
        self.prev_action = action
        
        print("ACTION: "+str(action))
        if action <= 0.5:
            return  self.defect()
        else:
            return  self.cooperate()

    #| 0   | defect  |
    #| 1   | cooperate |
    @classmethod
    def cooperate(self, opponent: Player) -> Action:
        return C

    @classmethod
    def defect(self, opponent: Player) -> Action:
        return D
