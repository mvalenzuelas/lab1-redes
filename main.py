import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import scipy.fft
import scipy.io.wavfile as waves
from scipy import signal
from scipy.io import wavfile


def reed_audio(file_name):
    """
    Function that allows you to read a .wav audio file and obtain the amplitude of the audio with respect to time
    :param file_name: String indicating the name of the .wav audio file. The extension must be included in the name
    :return: numerical vector with the amplitude of the audio signal in db (vector_audio) and sampling frequency
    with which the audio is read (sample_frequency)
    """
    sample_frequency, data = waves.read(file_name)
    try:
        vector_audio = data[:, 0]
    except IndexError:
        vector_audio = data

    return vector_audio, sample_frequency


def fourier_transform(vector):
    """
     Function that allows to calculate the fourier transform of an audio vector
     :param vector: A numeric vector containing the amplitude of an audio signal over time
     :return: A numeric vector containing the amplitude of the fourier transform with respect to the frequency
     """
    trans_vector = scipy.fft.fft(vector)
    return trans_vector


def inverse_fourier_transform(vector_transformada):
    """
    Function that allows to calculate the inverse of the fourier transform of the fourier transform
    :param vector_transformada: A numeric vector containing the amplitude of fourier transform with respect to the
    frequency
    :return: A numeric vector containing the amplitude if the inverse of the fourier transform with respect of time
    """
    trans_inv_vector = scipy.fft.ifft(vector_transformada).real
    return trans_inv_vector


def analyze(vector_audio, sample_frequency):
    """
     Function that allows generating graphs that show the amplitude of the audio signal with respect to time, the
     fourier transform of the audio signal, the inverse of the fourier transform, and the spectrogram of the signal
     audio
     :param vector_audio: numeric vector containing the amplitude of the audio signal with respect to time
     :param sample_frequency:integer value that represent the sampling frequency with which the audio is read
     """
    plt.subplots(2, 2, constrained_layout=True, figsize=(12, 8))
    len_vector_audio = len(vector_audio)

    plt.subplot(2, 2, 1)
    time = np.arange(0, len_vector_audio) / sample_frequency
    plt.plot(time, vector_audio)
    plt.title("Amplitud del vector en el tiempo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    trans_vector = fourier_transform(vector_audio)
    frequencies = np.arange(-len_vector_audio // 2, len_vector_audio // 2) * (sample_frequency / len_vector_audio)
    plt.subplot(2, 2, 2)
    plt.plot(frequencies, trans_vector.real)
    plt.title("Transformada de Fourier F(\u03C9)")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")
    plt.legend(("Reales", "Imaginarios"))

    trans_inversa = inverse_fourier_transform(trans_vector)
    plt.subplot(2, 2, 3)
    plt.plot(time, trans_inversa.real)
    plt.title("Transforma inversa de Fourier de F(\u03C9)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")

    plt.subplot(2, 2, 4)
    plt.specgram(vector_audio, Fs=sample_frequency, NFFT=649)
    plt.title("Espectrograma de la señal")
    plt.xlabel("Tiempo")
    plt.ylabel("Frecuencia")
    plt.show()


def signal_filter(noise_frequency, data, a, b, n):
    """
     Function that allows filtering a noisy audio signal with a specific sample rate using
     butterworth of the bandpass type
     :param noise_frequency: numeric value that represents the sampling frequency
     :param data: numeric vector representing the amplitude of the noisy audio signal over time
     :param a: numeric value greater than zero that represents the relative cutoff frequency to filter the signal
     :param b: numeric value greater than "a" that represents the relative cutoff frequency to filter the signal
     :param n: Filter order
     :return:
     """
    wn1 = 2 * a / noise_frequency
    wn2 = 2 * b / noise_frequency
    b, a = signal.butter(n, [wn1, wn2], 'bandpass')  # PASO DE BANDA
    filter_data = signal.filtfilt(b, a, data)
    return filter_data


def compare_filters(signal_to_filter, frequency):
    """
     Function that graphs the behavior of the Fourier transform of an audio signal by applying different
     filters
     :param signal_to_filter: numeric vector representing the amplitude of the noisy audio signal over time
     :param frequency: numeric value that represents the sampling frequency
     """
    filter1 = fourier_transform(signal_filter(frequency, signal_to_filter, 100, 5000, 6))
    filter2 = fourier_transform(signal_filter(frequency, signal_to_filter, 100, 5000, 3))
    filter3 = fourier_transform(signal_filter(frequency, signal_to_filter, 500, 6000, 1))
    filter4 = fourier_transform(signal_filter(frequency, signal_to_filter, 1000, 12000, 1))
    filter5 = fourier_transform(signal_filter(frequency, signal_to_filter, 1000, 12000, 4))
    filter6 = fourier_transform(signal_filter(frequency, signal_to_filter, 10000, 20000, 1))
    plt.subplots(3, 2, constrained_layout=True, figsize=(16, 12))
    plt.subplot(3, 2, 1)

    len_signal = len(signal_to_filter)
    frequency_array = np.arange(-len_signal // 2, len_signal // 2) * (frequency / len_signal)
    plt.plot(frequency_array, filter1.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=100 w2=5000 n=6")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")

    plt.subplot(3, 2, 2)
    plt.plot(frequency_array, filter2.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=100 w2=5000 n=3")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")

    plt.subplot(3, 2, 3)
    plt.plot(frequency_array, filter3.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=500 w2=6000 n=1")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")

    plt.subplot(3, 2, 4)
    plt.plot(frequency_array, filter4.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=1000 w2=12000 n=4")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")

    plt.subplot(3, 2, 5)
    plt.plot(frequency_array, filter5.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=1000 w2=12000 n=1")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")

    plt.subplot(3, 2, 6)
    plt.plot(frequency_array, filter6.real)
    plt.title("Transformada de Fourier para filtro con parametros w1=10000 w2=20000 n=1")
    plt.ylabel("Amplitud de la transformada de fourier")
    plt.xlabel("Frecuencia")
    plt.show()


def add_signals(signal1, sm_signal1, signal2, sm_signal2):
    """
     Function that allows two audio signals to be added and allows resampling in the event that one signal is bigger
     than the other
     :param signal1: numeric vector that represents the amplitude of the first audio signal in time
     :param sm_signal1: numeric value representing the sample rate
     :param signal2: numeric vector that represents the amplitude of the second audio signal in time
     :param sm_signal2: numeric vector that represents the amplitude of the first audio signal in time
     :return: Numerical vector that represents the amplitude of an audio signal formed by adding two audio signals
     (add_audios) and the sample frequency of this signal (sample_frequency_add_audios)
     """
    if len(signal2) >= len(signal1):
        resampled_audio = signal.resample(signal2, len(signal1))
        add_audios = signal1 + resampled_audio
        sample_frequency_add_audios = sm_signal1
    else:
        resampled_audio = signal.resample(signal1, len(signal2))
        add_audios = resampled_audio + signal2
        sample_frequency_add_audios = sm_signal2
    return add_audios, sample_frequency_add_audios

def menu(audio0, sample_frequency0, audio1, sample_frequency1, summed_signals, sample_frequency_summed_signals):
    print("Laboratorio 1: ")
    print()
    print("1- Gráficas de la señal de audio de Maximiliano Valenzuela")
    print("2- Gráficas de la señal de audio de Héctor Ballesteros")
    print("3- Graficas de la señal de audio ruido azul")
    print("4- Graficas de la señal de audio ruidosa")
    print("5- Comparación de filtros")
    print("6- Salir del programa")
    print()
    while True:
        option = int(input("Ingrese una opción: "))
        if (option == 1):
                    # Graph the audio signals, showing their amplitude, Fourier transform, inverse Fourier transform and Spectrogram
            analyze(audio0, sample_frequency0)
        elif (option == 2):
            analyze(audio1, sample_frequency1)
        elif (option == 3):
            analyze(audio2, sample_frequency2)
        elif (option == 4):
                    # Graph the properties of se summed signal
            analyze(summed_signals, sample_frequency_summed_signals)
        elif (option == 5):
                    # Graph a comparison of the Fourier transforms obtained of applying different filters to the summed signal
            compare_filters(summed_signals, sample_frequency_summed_signals)
        elif (option == 6):
            break
        else:
            print("Ingrese una opcion valida: ")

if __name__ == '__main__':
    # Reed the audio signals with the name and rut of the student
    audio0, sample_frequency0 = reed_audio("audio Hector.wav")
    audio1, sample_frequency1 = reed_audio("audio Maximiliano.wav")

    # Reed the noise audio signal
    audio2, sample_frequency2 = reed_audio("Ruido Azul.wav")

    # Choose the second audio signal and add the noise signal
    summed_signals, sample_frequency_summed_signals = add_signals(audio1, sample_frequency1, audio2, sample_frequency2)

    # Write in a .wav file the summed signal
    wavfile.write('audioRuidoso.wav', sample_frequency_summed_signals, summed_signals.astype(np.int16))

    # Apply a filter to the summed signal and graph the properties of the signal generated
    filter_signal = signal_filter(sample_frequency_summed_signals, summed_signals, 100, 5000, 6)

    # Write in a .wav file the filtered signal
    wavfile.write('ruidoFiltrado.wav', sample_frequency_summed_signals, filter_signal.astype(np.int16))

    menu(audio0, sample_frequency0, audio1, sample_frequency1, summed_signals, sample_frequency_summed_signals)