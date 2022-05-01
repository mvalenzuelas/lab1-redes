import scipy.io.wavfile as waves
import scipy.fft
from scipy import signal
from scipy.io import wavfile


import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import scipy.io.wavfile as waves
from scipy.signal import butter, sosfilt
from scipy.interpolate import InterpolatedUnivariateSpline


# Función que permite leer un archivo de auido .wav
def leer_audio(nombre_archivo):
    frecuencia_muestreo, data = waves.read(nombre_archivo)
    print(data)
    try:
        vector_audio = data[:, 0]
    except IndexError:
        vector_audio = data

    return vector_audio, frecuencia_muestreo


# Función que permite calcular la trasnformada discreta de Fourier de un vector numerico
def transformada_fourier(vector):
    trans_vector = scipy.fft.fft(vector)

    return trans_vector


def transformada_inversa_fourier(vector_transformada):
    trans_inv_vector = scipy.fft.ifft(vector_transformada).real
    return trans_inv_vector


def analizar(vector_audio, frecuencia_muestreo):
    plt.subplots(2, 2, constrained_layout=True, figsize=(12, 8))
    largo_vector_audio = len(vector_audio)

    plt.subplot(2, 2, 1)
    time = np.arange(0, largo_vector_audio) / frecuencia_muestreo
    plt.plot(time, vector_audio)
    plt.title("Amplitud del vector del audio grabado en el tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    trans_vector = transformada_fourier(vector_audio)
    frecuencia = np.arange(-largo_vector_audio // 2, largo_vector_audio // 2) * (
            frecuencia_muestreo / largo_vector_audio)
    plt.subplot(2, 2, 2)
    plt.plot(frecuencia, trans_vector.real)
    plt.title("Transformada de Fourier del audio grabado F(\u03C9)")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")
    plt.legend(("Reales", "Imaginarios"))

    trans_inversa = transformada_inversa_fourier(trans_vector)
    plt.subplot(2, 2, 3)
    plt.plot(time, trans_inversa.real)
    plt.title("Transforma inversa de Fourier de F(\u03C9)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    plt.subplot(2, 2, 4)
    plt.specgram(vector_audio, Fs=frecuencia_muestreo)
    plt.title("Espectograma del audio grabado")
    plt.xlabel("Tiempo")
    plt.ylabel("Freceuncia")
    plt.show()
 

def ruidoFilted(frecuencia_ruido, data):
    wn1 = 2*100/frecuencia_ruido
    wn2 = 2*5000/frecuencia_ruido
    b, a = signal.butter(6, [wn1,wn2], 'bandpass')  #PASO DE BANDA
    filtedData = signal.filtfilt(b, a, data) 
    wavfile.write('ruidoFiltrado.wav',frecuencia_ruido,filtedData.astype(np.int16))
    frecuencia_ruido, filtedData = wavfile.read('ruidoFiltrado.wav')
    return frecuencia_ruido, filtedData


def addSignals(audio1, frecuencia_muestreo1, audio2, frecuencia_muestreo2):
    if len(audio2) >= len(audio1):
        audio_interpolado = signal.resample(audio2, len(audio1))
        suma_audios = audio1 + audio_interpolado
        frecuencia_muestreo_sumado = frecuencia_muestreo1
    else:
        audio_interpolado=signal.resample(audio1,len(audio2))
        suma_audios=audio_interpolado+audio2
        frecuencia_muestreo_sumado = frecuencia_muestreo2
    return suma_audios, frecuencia_muestreo_sumado, audio_interpolado



if __name__ == '__main__':

    audio0, frecuencia_muestreo0=leer_audio("audio Hector.wav")
    audio1, frecuencia_muestreo1=leer_audio("audio Maximiliano.wav")
    audio2, frecuencia_muestreo2=leer_audio("Ruido Azul.wav")
    analizar(audio0,frecuencia_muestreo0)
    analizar(audio1,frecuencia_muestreo1)
    analizar(audio2,frecuencia_muestreo2)
    summed_signals, sample_frequency_summed_signals, interpolated_signal=addSignals(audio1, frecuencia_muestreo1, audio2, frecuencia_muestreo2)
    analizar(summed_signals,sample_frequency_summed_signals)
    wavfile.write('audioRuidoso.wav',sample_frequency_summed_signals,summed_signals.astype(np.int16))
    frecuencia_filted, filtedData = ruidoFilted(sample_frequency_summed_signals, summed_signals)
    analizar(filtedData,frecuencia_filted)

