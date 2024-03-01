#Corto 1 - Frozen Lake con Q-Learning
# Integrantes: Mark Albrand, Melissa Pérez, Jimena Hernández

import gym
import numpy as np
from IPython.display import clear_output
import time 
from gym.envs.toy_text.frozen_lake import generate_random_map

random_map = generate_random_map(size=4, p=0.8)  # Generar agujeros aleatorios con probabilidad 0.8 de no ser un agujero

env = gym.make('FrozenLake-v1', is_slippery=True, desc=random_map) 

no_states = env.observation_space.n # Número de estados
no_actions = env.action_space.n # Número de acciones

Q = np.zeros((no_states, no_actions))

# Parámetros del algoritmo
iteraciones = 10000
alpha = 0.8
gamma = 0.95

print("Entreno del agente")
print ('Q-table antes:')
print(Q)
# Entrenamiento
for i in range(iteraciones):
    estado = env.reset()[0]  # Estado inicial
    finish = False  # Si no se ha llegado al estado final

    while not finish: 
        accion = np.argmax(Q[estado,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))  # Se elige la acción con mayor valor en la Q-table
        estado_nuevo, recompensa, finish, _, _ = env.step(accion)  # Se ejecuta la acción, y se obtiene el nuevo estado y la recompensa
        Q[estado, accion] = Q[estado, accion] + alpha * (recompensa + gamma * np.max(Q[estado_nuevo]) - Q[estado, accion])  # Actualización de la Q-table
        estado = estado_nuevo


print('Q-table después:')
print(Q)

# Test
print("Test del agente")

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human', desc=random_map)  # Modo humano para ver el entorno

state = env.reset()[0]
done = False
sequence = []

# Se ejecuta el agente en el entorno
while not done:
    if np.max(Q[state]) > 0:
      action = np.argmax(Q[state])
    else:
      action = env.action_space.sample()
    
    # Se añade la acción a la secuencia
    sequence.append(action)

    # Se ejecuta la acción
    new_state, reward, done, info, _ = env.step(action)

    state = new_state

    # Se muestra el entorno
    clear_output(wait=True)
    env.render()
    time.sleep(0.5)

print(f"Sequence = {sequence}")  # Secuencia de acciones realizadas por el agente para llegar al estado final