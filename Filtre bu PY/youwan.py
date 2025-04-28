import matplotlib.pyplot as plt
import numpy as np

Fe=200
dt=1/Fe
t = np.arange(256)*dt
x = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*5*t)
sp = np.fft.fft(x)
freq = np.fft.fftfreq(x.size,dt)
plt.plot(freq, np.abs(sp.real))
plt.xlim(0,100)
plt.show()

def passe_bas(fcoup):
    filtre=[0]*len(freq)
    for i in range(0,len(freq)//2):
        if freq[i]<=fcoup:
            filtre[i]=1
    return filtre
print(passe_bas(3))
plt.plot(freq,passe_bas(3))
plt.xlim(0,50)
plt.show()
plt.plot(freq,np.abs((sp*passe_bas(3)).real))
plt.xlim(0,100)
plt.show()
plt.plot(freq,np.fft.ifft((sp*passe_bas(3))))
plt.xlim(0)
plt.show()