import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# definition du signal
dt = 0.0001
T1 = 2
T2 = 10
t = np.arange(0, T1*T2, dt)
signal = 2*np.cos(2*np.pi/T1*t) + np.cos(2*np.pi/T2*t) + 4*np.cos(2*np.pi/T1*t) + 8*np.cos(5*np.pi/T2*t+3) 
x=t

# affichage du signal
plt.subplot(211)
plt.plot(t,signal)
plt.xlim(0)

# calcul de la transformee de Fourier et des frequences
def fourrier_transform(signal):
    fourier = np.fft.fft(signal)
    return np.abs(fourier)
    # affichage de la transformee de Fourier
def affiche_signal(signal,x):
    plt.subplot(212)
    plt.plot(x, signal, label="real")
    plt.xlim(0)
    plt.legend()

    plt.show()

def passe_bas(freqlim,signal,dt):
    filtre=[0]*signal.size
    freq = np.fft.fftfreq(signal.size, d=dt)
    for i in range(signal.size):
        if fourrier_transform(signal,dt)[0][i]*(20*np.log(1.12)-10*np.log(1+((2*np.pi*freq[i])/freqlim*2*np.pi)**2)) == 1:
            filtre[i]=1
    return affiche_signal(fourrier_transform(signal,dt)*filtre)

affiche_signal(fourrier_transform(signal), t)
