import os
import math
import time
# import subprocess
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor


class DataProcessor:
    def __init__(self, sampling_rate, wav2vec_model_path):
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)
        self._sampling_rate = sampling_rate

    def extract_feature_by_audio_path(self, audio_path):
        speech_array, sampling_rate = librosa.load(audio_path, sr=self._sampling_rate)
        input_value = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)
        return input_value


def prepare_audio_feature(wav_file_path, fps=30, sampling_rate=16000, wav2vec_model_path=None):
    data_preprocessor = DataProcessor(sampling_rate, wav2vec_model_path)

    input_value = data_preprocessor.extract_feature_by_audio_path(wav_file_path)
    seq_len = math.ceil(len(input_value)/sampling_rate*fps)
    return {
        "audio_feature": input_value,
        "seq_len": seq_len
    }


def prepare_audio_feature_from_audio_data(wav_data, temp_path, fps=30, sampling_rate=16000, wav2vec_model_path=None):
    data_preprocessor = DataProcessor(sampling_rate, wav2vec_model_path)
    timestamp = str(int(time.time()))
    wav_temp_path = os.path.join(temp_path, f'tmp_audio_{timestamp}.wav')
    with open(wav_temp_path, "wb") as temp_file:
        temp_file.write(wav_data)
    input_value = data_preprocessor.extract_feature_by_audio_path(wav_temp_path)
    os.remove(wav_temp_path)
    seq_len = math.ceil(len(input_value) / sampling_rate * fps)
    return {
        "audio_feature": input_value,
        "seq_len": seq_len
    }


# def get_audio(file, start_time=0, duration=0):
#     args = ["ffmpeg", "-v", "error", "-i", file]
#     if start_time > 0:
#         args += ["-ss", str(start_time)]
#     if duration > 0:
#         args += ["-t", str(duration)]
#     try:
#         res =  subprocess.run(args + ["-f", "wav", "-"],
#                               stdout=subprocess.PIPE, check=True).stdout
#     except subprocess.CalledProcessError as e:
#         return False
#     return res


# if __name__ == '__main__':
#     file_path = '/home/ubuntu/stable-paw-comfyui/input/audio/x.mp3'
#     speech_array, sampling_rate = librosa.load(file_path, sr=16000)
#     print(speech_array)
#     result = get_audio(file_path, 0, 1)
#     # print(result)
#     print(np.frombuffer(result[:400], dtype=np.float32))
    
    
