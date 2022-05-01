import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import scipy.io.wavfile as waves
from scipy.signal import butter, sosfilt
from scipy.interpolate import InterpolatedUnivariateSpline


# Función que permite leer un archivo de auido .wav
def leer_audio(nombre_archivo):
    frecuencia_muestreo, data = waves.read(nombre_archivo)
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


def interpol(audio1, frecuencia_muestreo1, audio2, frecuencia_muestreo2):
    time = np.arange(0, 1)
    suma_audios = []
    frecuencia_muestreo_sumado = 0
    tiempo_interpolacion = []

    if len(audio2) >= len(audio1):
        tiempo_interpolacion = np.arange(0, len(audio2)) / frecuencia_muestreo1
        audio_interpolado = InterpolatedUnivariateSpline(tiempo_interpolacion, audio2)(tiempo_interpolacion)
        for i in range(len(audio1)):
            suma_audios.append(audio1[i] + audio_interpolado[i])
        time = np.arange(0, len(audio1)) / frecuencia_muestreo1
        frecuencia_muestreo_sumado = frecuencia_muestreo1
    else:
        tiempo_interpolacion = np.arange(0, len(audio1)) / frecuencia_muestreo2
        audio_interpolado = InterpolatedUnivariateSpline(tiempo_interpolacion, audio1)(tiempo_interpolacion)
        for i in range(len(audio2)):
            suma_audios.append(audio1[i] + audio_interpolado[i])
        time = np.arange(0, len(audio2)) / frecuencia_muestreo2
        frecuencia_muestreo_sumado = frecuencia_muestreo2

    return suma_audios, frecuencia_muestreo_sumado, audio_interpolado


def filtro(data, cutoff1, fs, order=5):
    nyq = fs
    normal_cutoff1 = cutoff1 / nyq
    s= butter(order, normal_cutoff1, btype='highpass', analog=False,output='sos')
    y = sosfilt(s,data)
    return y


if __name__ == '__main__':
    audio1, frecuencia_muestreo1 = leer_audio("audio Maximiliano.wav")
    audio2, frecuencia_muestreo2 = leer_audio("Ruido Azul.wav")
    analizar(audio1, frecuencia_muestreo1)
    analizar(audio2, frecuencia_muestreo2)
    audio_conjunto, frecuencia_muestreo_conjunto, audio_interpolado = interpol(audio1, frecuencia_muestreo1, audio2,
                                                                               frecuencia_muestreo2)

    analizar(audio_conjunto,frecuencia_muestreo_conjunto)
    FFT_filtrado=filtro(transformada_fourier(audio_conjunto),1000,frecuencia_muestreo_conjunto,10)
    analizar(transformada_inversa_fourier(FFT_filtrado),frecuencia_muestreo_conjunto)



