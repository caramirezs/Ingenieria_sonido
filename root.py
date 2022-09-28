# librerias necesarias
import numpy as np
from sklearn.decomposition import FastICA, PCA
from scipy.io.wavfile import read, write

# ACTIVIDAD 0 FASE 0 #
Fs, data = read('sonidos/mezcla1.wav')
senal_observada_1 = data[:441000]
Fs, data = read('sonidos/mezcla2.wav')
senal_observada_2 = data[:441000]
Fs, data = read('sonidos/mezcla3.wav')
senal_observada_3 = data[:441000]
# ESTA LÍNEA GUARDA LOS DATOS DE LOS 3 AUDIOS EN UNA VARIABLE
senales_observadas = np.c_[senal_observada_1, senal_observada_2, senal_observada_3]
# ESTA LINEA ESTANDARIZA LOS DATOS
# senales_observadas /= senales_observadas.std(axis=0)

def audio_mezclas(i):
    from IPython.display import Audio
    Fs = 44100
    print('AUDIO: {}'.format(i))
    return Audio(senales_observadas.T[i - 1], rate=Fs)


# ---------- #

# ACTIVIDAD 0 FASE 2 #
Fs, data = read('sonidos/Flute10s.wav')
sonido_1 = data[:441000,0]
Fs, data = read('sonidos/Chello10s.wav')
sonido_2 = data[:441000,0]
Fs, data = read('sonidos/Violin10s.wav')
sonido_3 = data[:441000,0]
# ESTA LÍNEA GUARDA LOS DATOS DE LOS 3 AUDIOS EN UNA VARIABLE
senales = np.c_[sonido_1, sonido_2, sonido_3]
senales = np.array(senales,dtype='float')
senales /= senales.std(axis=0)


def matriz_mezcla(dtt, dtm):
    import numpy as np
    intensidad = list()
    for u_mezcla in dtm:
        renglon = list()
        for u_tonop in dtt:
            renglon.append(1 / ((u_mezcla[0] - u_tonop[0]) ** 2 + (u_mezcla[1] - u_tonop[1]) ** 2))
        intensidad.append(renglon)
    matriz = np.array(intensidad)
    signals_mixes = (matriz @ senales.T).T
    #signals_mixes /= signals_mixes.std(axis=0)
    return matriz, signals_mixes

def run_ubicaciones(dt, dtt, dtm):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    plt.style.use('seaborn')
    rec_max_x = math.ceil(np.amax(dt, axis=0)[0] * 1.15)
    rec_max_y = math.ceil(np.amax(dt, axis=0)[1] * 1.15)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca()
    rectangle = plt.Rectangle((0, 0), rec_max_x, rec_max_y,
                              fc='purple', alpha=0.3)
    ax.add_patch(rectangle)

    i = 1
    for punto in dtt:
        circle = plt.Circle(punto, radius=0.015 * min(np.amax(dt, axis=0)), fc='green')
        ax.add_patch(circle)
        plt.text(punto[0], punto[1], '$s_{}(t)$'.format(i), fontsize=20)
        i += 1

    i = 1
    for punto in dtm:
        circle = plt.Circle(punto, radius=0.015 * min(np.amax(dt, axis=0)), fc='b')
        ax.add_patch(circle)
        plt.text(punto[0], punto[1], '$x_{}(t)$'.format(i), fontsize=20)
        i += 1
        for punto2 in dtt:
            plt.plot([punto2[0], punto[0]], [punto2[1], punto[1]],
                     "--", color="g", lw=1)

    ax.set_xticks(np.arange(0, rec_max_x * 1.05, 0.5))
    ax.set_yticks(np.arange(0, rec_max_y * 1.05, 0.5))

    plt.axis('scaled')
    plt.show()
    return matriz_mezcla(dtt, dtm)

def graf_s(vector_senal, tono, segIni=1, segFin=1.05):
    import numpy as np
    from IPython.display import Audio
    import matplotlib.pyplot as plt
    locs = np.linspace(0, ((segFin - segIni) * 44100), 11)
    inicio = int(Fs * segIni)
    fin = int(Fs * segFin)
    fig, ax1 = plt.subplots(figsize=(12, 3))
    label = ['{} seg'.format(segIni), '', '', '', '', '{} seg'.format((segIni + segFin) / 2), '', '', '', '',
             '{} seg'.format(segFin)]
    ax1.plot(vector_senal.T[tono - 1][inicio:fin])
    plt.sca(ax1)
    plt.title('Mezcla: {}'.format(tono))
    plt.ylabel('Amplitud')
    plt.xticks(locs, label)
    plt.show()
    return Audio(vector_senal.T[tono - 1], rate=Fs)

def graf_s_2(vector_senal, tono, segIni=1, segFin=1.5):
    import numpy as np
    from IPython.display import Audio
    import matplotlib.pyplot as plt
    locs = np.linspace(0, ((segFin - segIni) * 44100), 11)
    inicio = int(Fs * segIni)
    fin = int(Fs * segFin)
    fig, ax1 = plt.subplots(figsize=(12, 3))
    label = ['{} seg'.format(segIni), '', '', '', '', '{} seg'.format((segIni + segFin) / 2), '', '', '', '',
             '{} seg'.format(segFin)]
    ax1.plot(vector_senal.T[tono - 1][inicio:fin])
    plt.sca(ax1)
    plt.title('Mezcla: {}'.format(tono))
    plt.ylabel('Amplitud')
    plt.xticks(locs, label)
    plt.show()
    return Audio(vector_senal.T[tono - 1], rate=Fs)


t = np.linspace(0, 10, 441000)
Fs = 44100


def tabla_mezclas(dtt, dtm):
    import pandas as pd
    matriz, signals_mixes = matriz_mezcla(dtt, dtm)
    return pd.DataFrame(np.array([t, signals_mixes.T[0], signals_mixes.T[1], signals_mixes.T[2]]).T,
                        columns=['TIEMPO', 'VALOR MEZCLA 1', 'VALOR MEZCLA 2', 'VALOR MEZCLA 3']).set_index('TIEMPO')