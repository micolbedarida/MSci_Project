import numpy as np
import matplotlib.pyplot as plt

# defining constants for use
h=6.62607004e-34
c=299792458.
k=1.38064852e-23

# variables
Ts = [300,310,320,330,340]
wns = np.linspace(100,700,1000)
nu = wns*c*100

# defining functions
def planck(T, nu):
    return 2.*h*(np.power(nu,3))/(c**2 * (np.exp(h*nu/(k*T)) -1.))

planck_curve = np.vectorize(planck)

# plotting
for T in Ts:
    plt.plot(wns, planck_curve(T, nu), label=("Temperature = " + str(T) + "K"))

plt.ylabel('Spectral radiance (kW$\cdot$sr$^{-1}\cdot$m$^{-2}\cdot$nm$^{-1}$)', size=18)
plt.xlabel('Wavenumber (cm$^{-1}$)', size=18)
plt.tick_params(labelsize=18)
plt.legend(prop={'size': 18})
plt.grid()