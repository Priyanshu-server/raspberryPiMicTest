import numpy as np
import librosa
from allvarsconfig import *
import librosa.display
import matplotlib.pyplot as plt


def extract_features(data):
  result = np.array([])
  #ZCR
  zcr = librosa.feature.zero_crossing_rate(y=data)
  mean_zcr = np.mean(zcr.T,axis = 0)
  result = np.hstack((result,mean_zcr))
  
  #Croma STFT
  stft = np.abs(librosa.stft(data))
  chrome_stft=librosa.feature.chroma_stft(S=stft, sr=SAMPLE_RATE)
  mean_chroma_stft = np.mean(chrome_stft.T, axis=0)
  result = np.hstack((result, mean_chroma_stft))

  #MFCC
  mfcc = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE)
  mean_mfcc = np.mean(mfcc.T,axis = 0)
  result = np.hstack((result, mean_mfcc))
  
  # Root Mean Square Value
  rms = librosa.feature.rms(y=data)
  mean_rms = np.mean(rms.T,axis = 0)
  result = np.hstack((result, mean_rms))

  # MelSpectogram
  mel = librosa.feature.melspectrogram(y=data, sr=SAMPLE_RATE)
  mean_mel = np.mean(mel.T,axis = 0)
  result = np.hstack((result, mean_mel))

  return result