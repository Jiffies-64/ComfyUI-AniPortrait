import os
import math

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

    def extract_feature_from_audio_data(self, audio_data):
        input_value = np.squeeze(self._processor(audio_data, sampling_rate=self._sampling_rate).input_values)
        return input_value


def prepare_audio_feature(wav_file_path, fps=30, sampling_rate=16000, wav2vec_model_path=None):
    data_preprocessor = DataProcessor(sampling_rate, wav2vec_model_path)

    input_value = data_preprocessor.extract_feature_by_audio_path(wav_file_path)
    seq_len = math.ceil(len(input_value)/sampling_rate*fps)
    return {
        "audio_feature": input_value,
        "seq_len": seq_len
    }


def prepare_audio_feature_from_audio_data(wav_data, fps=30, sampling_rate=16000, wav2vec_model_path=None):
    data_preprocessor = DataProcessor(sampling_rate, wav2vec_model_path)

    input_value = data_preprocessor.extract_feature_from_audio_data(wav_data)
    seq_len = math.ceil(len(input_value)/sampling_rate*fps)
    return {
        "audio_feature": input_value,
        "seq_len": seq_len
    }


