import matplotlib.pyplot as plt
import numpy as np

Fe=200
dt=1/Fe
t = np.arange(256)*dt
x = np.sin(2*np.pi*2*t) + np.sin(2*np.pi*5*t)
sp = np.fft.fft(x)
freq = np.fft.fftfreq(x.size,dt)
Signalsd=[]

def affiche_courbe(signal, abscisse, borne_inf = None, borne_sup = None):   #Afficher une courbe
    plt.plot(abscisse, signal)
    plt.xlim(borne_inf,borne_sup)
    plt.show()

def passe_bas(fcoup):                                                   #Filtre passe bas
    filtre=[0]*len(freq)
    for i in range(0,len(freq)//2):
        if freq[i]<=fcoup:
            filtre[i]=1
    return filtre

def affiches_courbes(Signals):                                          #Afficher plusieurs courbes sur une seule fenêtre
    for i in range(len(Signals)):
        plt.subplot(len(Signals),1,i+1)
        plt.plot(Signals[i][1],Signals[i][0])
        plt.xlim(Signals[i][2],Signals[i][3])
    plt.show()

def affiches_courbes_dico(Signals):                                          #Afficher plusieurs courbes sur une seule fenêtre
    for i in range(len(Signals)):
        plt.subplot(len(Signals),1,i+1)
        plt.plot(Signals[i]["abscisse"],Signals[i]["signal"])
        plt.xlim(Signals[i]["borne_inf"],Signals[i]["borne_sup"])
    plt.show()

def traitement_signal(signal, freq_coupure):
    Signal = {"signal" : signal, "abscisse" : t, "borne_inf" : 0, "borne_sup" : None}
    fft = {"signal" : np.abs(np.fft.fft(signal.real)), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 100}
    Filtre = {"signal" : passe_bas(freq_coupure), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 50}
    Fft_filtré = {"signal" : np.abs((signal*passe_bas(freq_coupure)).real), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : 100}
    Signal_filtré = {"signal" : np.fft.ifft((sp*passe_bas(freq_coupure))), "abscisse" : np.fft.fftfreq(signal.size,dt), "borne_inf" : 0, "borne_sup" : None}
    Signalsd.append(Signal)
    Signalsd.append(fft)
    Signalsd.append(Filtre)
    Signalsd.append(Fft_filtré)
    Signalsd.append(Signal_filtré)

    affiches_courbes_dico(Signalsd)

Signals=[(np.abs(sp.real), freq,0,100),(passe_bas(3), freq,0,50),(np.abs((sp*passe_bas(3)).real), freq,0,100),(np.fft.ifft((sp*passe_bas(3))), freq,0,None)]
print(passe_bas(3))

affiche_courbe(np.abs(sp.real), freq,0,100)
affiche_courbe(passe_bas(3), freq,0,50)
affiche_courbe(np.abs((sp*passe_bas(3)).real), freq,0,100)
affiche_courbe(np.fft.ifft((sp*passe_bas(3))), freq,0)

affiches_courbes(Signals)

traitement_signal(x,3)