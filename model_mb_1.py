'''
1 0 0 0 0
0 1 0 0 0
0 0 1 0 0 
0 0 0 1 0 
0 0 0 0 1
'''

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import namedtuple

import two_step_task as ts            
import analysis as an

one_hot = keras.utils.to_categorical
sse_loss = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

Episode = namedtuple('Episode', ['states', 'rewards', 'actions', 'pfc_inputs', 'pfc_states', 'pred_states','task_rew_states', 'n_trials'])

default_params = {
    'n_episodes'  : 500,                                           
    'episode_len' : 250,                                       
    'max_step_per_episode' : 600,
    'gamma' : 0.9,        

    'n_back': 20,
    'n_pfc' : 16,  
    'pfc_learning_rate' : 0.01,

    'n_str' : 10, 
    'str_learning_rate' : 0.05,
    'entropy_loss_weight' : 0.05}


def run_simulation(save_dir = None, pm = default_params):
    np.random.seed(int.from_bytes(os.urandom(4), 'little'))         # Random seed for each agent                     
    task = ts.Two_Step()                                            # Initialize the environment

    # PFC Model 
    pfc_input_layer = layers.Input(shape=(pm['n_back'], task.n_states+task.n_actions))                      # n_back x n_state + n_actions + n_str
    pfc_input_buffer = np.zeros([pm['n_back'], task.n_states+task.n_actions], dtype = np.float32)

    rnn = layers.GRU(pm['n_pfc'], unroll=True, name='rnn')(pfc_input_layer)                                             # Recurrent layer.              
    state_pred = layers.Dense(task.n_states, activation='softmax', name='state_pred')(rnn)                              # Output layer predicts next state
    action_logits = layers.Dense(task.n_actions, activation = None, name = 'action_logits')(rnn)                        # Output layer predicts next action
    
    PFC_model = keras.Model(inputs=pfc_input_layer, outputs=[state_pred, action_logits])                                # The complete PFC model
    pfc_optimizer = keras.optimizers.Adam(learning_rate=pm['pfc_learning_rate'])        
    Get_pfc_state = keras.Model(inputs=PFC_model.input, outputs=PFC_model.get_layer('rnn').output)                      # PFC model gives RNN output 

    def update_pfc_input(a,s):
        '''Update the inputs to the PFC network given the previous action, state, and striatal layer'''
        pfc_input_buffer[:-1,:] = pfc_input_buffer[1:,:]
        pfc_input_buffer[-1,:] = 0 
        
        pfc_input_buffer[-1,s] = 1               # One hot encoding of state.
        pfc_input_buffer[-1,a+task.n_states] = 1 # One hot encoding of action.
    
    def get_masked_PFC_inputs(pfc_inputs):
        '''Return array of PFC input history with the most recent state masked, 
        used for training as the most recent state is the prediction target.'''
        masked_pfc_inputs = np.array(pfc_inputs)
        masked_pfc_inputs[:,-1,:task.n_states] = 0
        return masked_pfc_inputs

    # Striatum model
    obs_state = layers.Input(shape=(task.n_states,))                                        # Observable state features
    pfc_state = layers.Input(shape=(pm['n_pfc'],))                                          # PFC activity features
    combined_features = keras.layers.Concatenate(axis=1)([obs_state, pfc_state])
    relu = layers.Dense(pm['n_str'], activation="relu")(combined_features)
    
    actor = layers.Dense(task.n_actions, activation="softmax")(relu)
    critic = layers.Dense(1)(relu)
    
    Str_model = keras.Model(inputs=[obs_state, pfc_state], outputs=[actor, critic])         # Full model
    str_optimizer = keras.optimizers.Adam(learning_rate=pm['str_learning_rate'])


    def store_trial_data(s, r, a, pfc_s, V):
        'Store state, reward and subseqent action, PFC input buffer, PFC activity, value.'
        states.append(s)
        rewards.append(r)
        actions.append(a)
        pfc_inputs.append(pfc_input_buffer.copy())
        pfc_states.append(pfc_s)
        values.append(V)
        task_rew_states.append(task.high_state)

    # Running the model
    
    s = task.reset()        # State set to "choice"
    r = 0
    pfc_s = Get_pfc_state.predict_on_batch(pfc_input_buffer[np.newaxis,:,:])

    episode_buffer = []
        
    for e in range(pm['n_episodes']):
        
        step_n = 0                          
        start_trial = task.trial_n
        
        # Episode history variables
        states  = []       # int
        rewards = []       # float
        actions = []       # int
        pfc_inputs = []    # (1,30,n_states+n_actions)
        pfc_states = []    # (1,n_pfc)
        values = []        # float
        task_rew_states = [] # bool

        while True:
            step_n += 1

            state_input = one_hot(s, task.n_states)[None,:] 

            action_probs, V = Str_model([state_input, pfc_s])
            a = np.random.choice(task.n_actions, p =np.squeeze(action_probs))

            store_trial_data(s, r, a, pfc_s, V)  # Please store my step data 
            s, r = task.step(a)                             # Get new reward and state 

            update_pfc_input(a,s)                # Update the pfc buffer 
            pfc_s = Get_pfc_state.predict_on_batch(pfc_input_buffer[np.newaxis,:,:])                    # Get the new pfc state
    
            n_trials = task.trial_n - start_trial
            if n_trials == pm['episode_len'] or step_n >= pm['max_step_per_episode'] and s == 0:
                break

        state_pred, _ = PFC_model(get_masked_PFC_inputs(pfc_inputs))                                    # Compute all predicted state probs 
        pred_states = np.argmax(state_pred, axis=1)                                                     # Turn probs to predicted state
        episode_buffer.append(Episode(np.array(states), np.array(rewards), np.array(actions), np.array(pfc_inputs),     # Store episodal data
                               np.vstack(pfc_states), np.array(pred_states), np.array(task_rew_states), n_trials))


        returns = np.zeros([len(rewards),1], dtype='float32')               # Create array to store the return for every time step in the episode
        returns[-1] = V                                                     # Make the final return by the last critic output V 
        for i in range(1, len(returns)):                        
            returns[-i-1] = rewards[-i] + pm['gamma']*returns[-i]           # Compute the rest of the returns 
                 
        advantages = (returns - np.vstack(values)).squeeze()                # Compute advantages, "How much reward actually obtained 
        eps = 1e-8                                                          #                      from this timestep onwards" 

        with tf.GradientTape() as str_tape: 
            action_probs_g, values_g = Str_model([one_hot(states, task.n_states), np.vstack(pfc_states)])               # Run entire episode thru striatum
            critic_loss = sse_loss(values_g, returns)                                                                   # Set the critic loss                            
            
            log_chosen_probs = tf.math.log(tf.gather_nd(action_probs_g, [[i,a] for i,a in enumerate(actions)]) + eps)   # Get log probabilitity of actions chosen
            entropy = -tf.reduce_sum(action_probs_g*tf.math.log(action_probs_g + eps),1)                                # Get the entropy term 
            actor_loss = tf.reduce_sum(-log_chosen_probs*advantages-entropy*pm['entropy_loss_weight'])                  # Set the actor loss 

            grads = str_tape.gradient(actor_loss+critic_loss, Str_model.trainable_variables)                            # Compute gradients
    
        str_optimizer.apply_gradients(zip(grads, Str_model.trainable_variables))                                        # Apply gradients to update weights


        with tf.GradientTape() as pfc_tape:
            state_pred, action_logits = PFC_model(get_masked_PFC_inputs(pfc_inputs))
            state_loss = sse_loss(state_pred, one_hot(states, task.n_states))                                           # Set the state loss
            
            action_probs = tf.nn.softmax(action_logits)
            log_chosen_probs = tf.math.log(tf.gather_nd(action_probs, [[i,a] for i, a in enumerate(actions)]) + eps)    # Get log probability of actions chosen
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + eps), axis = 1)                          # Get the entropy term
            pfc_actor_loss = tf.reduce_sum(-log_chosen_probs * advantages - entropy * pm['entropy_loss_weight'])        # Set the actor loss

            total_loss = state_loss +  pfc_actor_loss                                                                   # Combine the losses 
            pfc_grads = pfc_tape.gradient(total_loss, PFC_model.trainable_variables)                                    # Compute gradients

        pfc_optimizer.apply_gradients(zip(pfc_grads, PFC_model.trainable_variables))                                    # Apply gradients to update weights
        
        if e == 0:
            print('Model MB. Striatum Chooses.\n')
            print('PFC has *silent* actor\n')
            print('PFC receive 5 states as part of vector length 5\n')

        if save_dir:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            with open(os.path.join(save_dir,'params.json'), 'w') as fp:
                json.dump(pm, fp, indent=4)
            with open(os.path.join(save_dir, 'episodes.pkl'), 'wb') as fe: 
                pickle.dump(episode_buffer, fe)
            PFC_model.save(os.path.join(save_dir, 'PFC_model.keras'))
            Str_model.save(os.path.join(save_dir, 'Str_model.keras'))