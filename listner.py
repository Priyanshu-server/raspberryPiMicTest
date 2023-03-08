import pyaudio
import wave
import librosa
import numpy as np
import matplotlib.pyplot as plt



class Listner(object):
    def __init__(self,fo,channels,sample_rate,chunk_size,input = True):
        self.format = fo
        self.channels = channels
        self.input = input
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def _stop_rec(self,stream,audio):
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Finished recording.")

    def _stream_init(self,audio):
        stream = audio.open(format=self.format, channels=self.channels,
                        rate=self.sample_rate, input=self.input,
                        frames_per_buffer=self.chunk_size)
        return stream
    
    def _buffer_to_float(self,frames,n_bytes):
        dtype = np.float32
        audio_data = librosa.util.buf_to_float(b''.join(frames), n_bytes=4, dtype=dtype)
        return audio_data
    
    def _plot_wave(self,data):
        plt.plot(data)
        plt.show()
    
    def listen(self,seconds,show = False,save = False):
        audio = pyaudio.PyAudio()
        # start recording
        stream = self._stream_init(audio)
        print("Recording...")

        frames = []
        for _ in range(0, int(self.sample_rate / self.chunk_size * seconds)):
            data = stream.read(self.chunk_size)
            frames.append(data)

        # stop recording
        self._stop_rec(stream,audio)

        # Converting captured data into right format
        audio_data = self._buffer_to_float(frames,4)

        if show:
            self._plot_wave(audio_data)
        
        print("Audio Data Length : ",len(audio_data))

        if save:
            waveFile = wave.open("output.wav", 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(audio.get_sample_size(self.format))
            waveFile.setframerate(self.sample_rate)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()

        return audio_data
    
    def noise(self,data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self,data, rate=0.8):
        return librosa.effects.time_stretch(y = data, rate = rate)

    def shift(self,data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self,data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(y = data, n_steps = pitch_factor, sr = self.sample_rate)



# write the audio data to a WAV file
'''
# waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
# waveFile.setnchannels(CHANNELS)
# waveFile.setsampwidth(audio.get_sample_size(FORMAT))
# waveFile.setframerate(RATE)
# waveFile.writeframes(b''.join(frames))
# waveFile.close()
'''
