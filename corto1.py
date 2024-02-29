import gym
import random
import numpy as np
from IPython.display import clear_output
import time 

env = gym.make('FrozenLake-v1', is_slippery=True)

no_states = env.observation_space.n
no_actions = env.action_space.n

Q = np.zeros((no_states, no_actions))

# Parámetros del algoritmo
iteraciones = 100000
alpha = 0.8
gamma = 0.95

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
        accion = np.argmax(Q[estado,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    
        estado_nuevo, recompensa, finish, _, _ = env.step(accion)

        Q[estado, accion] = Q[estado, accion] + alpha * (recompensa + gamma * np.max(Q[estado_nuevo]) - Q[estado, accion])
        estado = estado_nuevo


print('Q-table después:')
print(Q)

# Test
print("Test del agente")

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

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