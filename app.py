from listner import Listner
import pyaudio
from allvarsconfig import *
from audiofeat import extract_features
import numpy as np
from collections import defaultdict

import tensorflow as tf
import time 
import pandas as pd


connections_arr = []
counter = 0

# try to predict with all the sounds and use the combined probabilities
def get_all_features(audio_data,noisy_data = None,stretched_pitched_data = None ):
    #getting features
    #Original Data
    res_1 = extract_features(audio_data)
    result = np.array(res_1)

    #Noisy Data
    if noisy_data is not None:
        res_2 = extract_features(noisy_data)
        result = np.vstack((result,res_2))

    #Stratching and Pitching
    if stretched_pitched_data is not None:
        res_3 = extract_features(stretched_pitched_data)
        result = np.vstack((result,res_3))
    return result


# Driver Code
if __name__ == "__main__":
    print("Loading Model :: ")
    model = tf.keras.models.load_model("model.h5")
    categorical = {0:'happy',1:'sad',2:'angry',3:'disgust',
               4: 'fear',5:'neutral',6:'surprise'}

    #Listining to Sound
    while True:
        l = Listner(pyaudio.paInt16,CHANNELS,SAMPLE_RATE,CHUNK,True)
        audio_data = l.listen(2.5,show = False,save=True)
        noisy_audio_data = l.noise(audio_data)
        stretched_audio_data = l.stretch(audio_data)
        stretch_pitched_audio_data = l.pitch(stretched_audio_data,SAMPLE_RATE)

        all_feat = get_all_features(audio_data,noisy_audio_data,stretch_pitched_audio_data)
        
        predictions = []
        idx_count = defaultdict(lambda : 0)

        for i in range(all_feat.shape[0]):
            print(f"Prediction - {i}")
            pred = model.predict(all_feat[i][tf.newaxis,...])
            print("SUM : ",tf.math.reduce_sum(pred,axis = -1))
            predictions.append(pred)
            true_idx = tf.math.argmax(pred,axis = -1).numpy()[0]
            idx_count[true_idx] = idx_count[true_idx] + 1

            connections_arr.append(pred[0])

        print("------------------- Dict ------------")
        for i,j in idx_count.items():
            print("{} : {}".format(i,j))

        print("_______________---")
        is_predicted = False
        true_idx = None
        for item,val in idx_count.items():
            if val>1:
                print("True IDX : ",item)
                true_idx = item
                is_predicted = True
                print("-"*40)
        
        if not is_predicted:
            true_idx = tf.math.argmax(predictions[0],axis = -1).numpy()[0]
            print("True Index :: ",true_idx)


        print("CATEGORY ::: ",categorical[true_idx])

        time.sleep(5)
        counter += 1


        if counter == 25:
            print("SHAPE :: ",np.asarray(connections_arr).shape)
            df = pd.DataFrame(data = connections_arr,columns = categorical.values())
            df.to_csv('raw_data.csv', index=False)
            


