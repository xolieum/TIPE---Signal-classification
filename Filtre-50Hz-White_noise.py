import matplotlib.pyplot as plt
import numpy as np

mean = 0
std = 1 
num_samples = 1000
x = np.random.normal(mean, std, size=num_samples)
Fe=200
dt=1/Fe
t = np.arange(1000)*dt
sp = np.fft.fft(x)
freq = np.fft.fftfreq(x.size,dt)

def passe_bas(fcoup):                                                   #Filtre passe bas
    filtre=[0]*len(freq)
    for i in range(0,len(freq)//2):
        if freq[i]<=fcoup:
            filtre[i]=1
    return filtre

def affiches_courbes_dico(Signals):                                          #Afficher plusieurs courbes sur une seule fenêtre
    for i in range(len(Signals)):
        plt.subplot(len(Signals),1,i+1)
        plt.plot(Signals[i]["abscisse"],Signals[i]["signal"])
        plt.xlim(Signals[i]["borne_inf"],Signals[i]["borne_sup"])
    plt.show()

def traitement_signal(signal, freq_coupure):
    Signals=[]
    Signal = {"signal" : signal, "abscisse" : t, "borne_inf" : 0, "borne_sup" : None}
    fft = {"signal" : np.abs(np.fft.fft(signal.real)), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 200}
    Filtre = {"signal" : passe_bas(freq_coupure), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 100}
    Fft_filtré = {"signal" : np.abs((signal*passe_bas(freq_coupure)).real), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 200}
    Signal_filtré = {"signal" : np.fft.ifft((sp*passe_bas(freq_coupure))), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : None}
    Signals.append(Signal)
    Signals.append(fft)
    Signals.append(Filtre)
    Signals.append(Fft_filtré)
    Signals.append(Signal_filtré)

    affiches_courbes_dico(Signals)

traitement_signal(x,50)