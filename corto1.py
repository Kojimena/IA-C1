import gym
import random
import numpy as np
from IPython.display import clear_output
import time 

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

no_states = env.observation_space.n
no_actions = env.action_space.n

Q = np.zeros((no_states, no_actions))

# Parámetros del algoritmo
iteraciones = 1000
alpha = 0.5
gamma = 0.9

print("Entreno del agente")
print ('Q-table antes:')
print(Q)
# Entrenamiento
for i in range(iteraciones):
    estado = env.reset()[0]
    finish = False

    while not finish:
        # estado viene en este formato: (0, {'prob': 1})
        # elegir la acción con mayor valor en el estado actual
        if np.max(Q[estado]) > 0:
            accion = np.argmax(Q[estado])
            
        else:
            accion = env.action_space.sample()

        estado_nuevo, recompensa, finish, info, _ = env.step(accion)

        Q[estado, accion] = Q[estado, accion] + alpha * (recompensa + gamma * np.max(Q[estado_nuevo]) - Q[estado, accion])
        estado = estado_nuevo


print('Q-table después:')
print(Q)

# Test
print("Test del agente")

state = env.reset()[0]
done = False
sequence = []

while not done:
    if np.max(Q[state]) > 0:
      action = np.argmax(Q[state])

    else:
      action = env.action_space.sample()
    
    sequence.append(action)

    new_state, reward, done, info, _ = env.step(action)

    state = new_state

    clear_output(wait=True)
    env.render()
    time.sleep(0.5)

print(f"Sequence = {sequence}")