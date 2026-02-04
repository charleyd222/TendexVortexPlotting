import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

with open('dataSim4', 'rb') as f:
    data = pickle.load(f)

fig, ax = plt.subplots(1)
fig.suptitle('1: Mass Quadropole 2: Current Quadropole')

model_param = data['param']
ax.set_title(r'$\Omega_1 =$ %s. $\Omega_2 =$ %s. $C_2 =$ %s. $\omega_2 =$ %s$\pi$' % (model_param.omega1, model_param.omega2, model_param.C2, int((model_param.w2 / np.pi ) * 100) / 100))


def theta_by_phi(data):
    sc = ax.scatter(data['x'], data['y'], s=2, c=data['t'], cmap='viridis')

    fig.colorbar(sc, label='t')
    ax.set_xlim(0,2*np.pi)
    ax.set_xlabel('phi')
    ax.set_ylim(0,np.pi)
    ax.set_ylabel('theta')

def test(data):
    d = {}
    for i in range(len(data['t'])):
        d[data['t'][i]] = data['x'][i]

def phi_by_t(data):
    ax.scatter(data['t'], data['y'], s=2)

    ax.set_xlim(0,int(max(data['t'])+1))
    ax.set_xlabel('t')
    ax.set_ylim(0,np.pi)
    ax.set_ylabel('theta')

phi_by_t(data)
plt.show()