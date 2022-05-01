import scipy.io.wavfile as waves
import scipy.fft
from scipy import signal

import matplotlib.pyplot as plt
import numpy as np


# Función que permite leer un archivo de auido .wav
def leer_audio(nombre_archivo):
    frecuencia_muestreo, data = waves.read(nombre_archivo)
    try:
        vector_audio = data[:, 0]
    except IndexError:
        vector_audio=data

    return vector_audio, frecuencia_muestreo


# Función que permite graficar en funcioón del tiempo, la amplitud de un vector de audio proveniente de un archivo .wav



# Función que permite calcular la trasnformada discreta de Fourier de un vector numerico
def transformada_fourier(vector, largo_vector, frecuencia_muestreo):
    trans_vector =scipy.fft.fft(vector)
    frecuencia = np.arange(-largo_vector // 2, largo_vector // 2) * (frecuencia_muestreo / largo_vector)

    return trans_vector, frecuencia


def transformada_inversa_fourier(vector_transformada, largo_vector_audio, frecuencia_muestreo):

    trans_inv_vector = scipy.fft.ifft(vector_transformada).real

    return trans_inv_vector


def analizar(vector_audio,frecuencia_muestreo):
    vector_audio, frecuencia_muestreo
    plt.subplots(2, 2, constrained_layout=True,figsize=(12,8))
    largo_vector_audio = len(vector_audio)
    plt.subplot(2,2,1)
    time = np.arange(0, largo_vector_audio) / frecuencia_muestreo
    plt.plot(time, vector_audio)
    plt.title("Amplitud del vector del audio grabado en el tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    trans_vector, frecuencia = transformada_fourier(vector_audio, largo_vector_audio, frecuencia_muestreo)
    plt.subplot(2,2,2)
    plt.plot(frecuencia, trans_vector.real)
    plt.title("Transformada de Fourier del audio grabado F(\u03C9)")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")
    plt.legend(("Reales", "Imaginarios"))


    trans_inversa = transformada_inversa_fourier(trans_vector, largo_vector_audio, frecuencia_muestreo)
    plt.subplot(2,2,3)
    plt.plot(time, trans_inversa.real)
    plt.title("Transforma inversa de Fourier de F(\u03C9)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")


    plt.subplot(2,2,4)
    plt.specgram(vector_audio,Fs=frecuencia_muestreo)
    plt.title("Espectograma del audio grabado")
    plt.xlabel("Tiempo")
    plt.ylabel("Freceuncia")
    plt.show()


if __name__ == '__main__':
    audio0, frecuencia_muestreo0=leer_audio("audio Hector.wav")
    audio1,frecuencia_muestreo1=leer_audio("audio Maximiliano.wav")
    audio2,frecuencia_muestreo2=leer_audio("Ruido Azul.wav")

    analizar(audio0,frecuencia_muestreo0)
    analizar(audio1,frecuencia_muestreo1)
    analizar(audio2,frecuencia_muestreo2)


