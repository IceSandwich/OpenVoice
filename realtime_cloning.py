# -*- coding: utf-8 -*-
import pyaudio
import numpy as np
import torch
import argparse
import enum
from openvoice.api import ToneColorConverter

@enum.unique
class AudioStreamType(enum.Enum):
    Input = 0
    Output = 1

class AudioSteam:
    instance = pyaudio.PyAudio()

    def __init__(self, sampling_rate: int, buffer_size: int, type: AudioStreamType) -> None:
        self.sampling_rate = sampling_rate
        self.type = type
        self.buffer_size = buffer_size

    def __enter__(self):
        print(f"> Opening {self.type} stream in sampling rate {self.sampling_rate} and buffer size {self.buffer_size}...")
        self.stream = AudioSteam.instance.open(
            format=pyaudio.paFloat32,
            channels=1, #单声道
            rate=self.sampling_rate,
            input=(self.type == AudioStreamType.Input),
            output=(self.type == AudioStreamType.Output),
            frames_per_buffer=self.buffer_size,
            #input_device_index=(13 if self.type == AudioStreamType.Input else None),
            #output_device_index=(1 if self.type == AudioStreamType.Output else None),
        )
        return self

    def __exit__(self):
        print(f"> Closing stream {self.type}...")
        self.stream.stop_stream()
        self.stream.close()

    def read(self, buffer_size: int = -1) -> np.ndarray:
        if buffer_size == -1:
            buffer_size = self.buffer_size
        
        data = self.stream.read(buffer_size)
        return np.frombuffer(data, dtype=np.float32)

    def write(self, data: bytes):
        self.stream.write(data)

def main(converter: ToneColorConverter, src_tone: torch.tensor, dst_tone: torch.tensor):
    sampling_rate = converter.hps.data.sampling_rate
    filter_length = converter.hps.data.filter_length
    hop_length = converter.hps.data.hop_length
    win_length = converter.hps.data.win_length
    tau = 0.3
    print("> Use sampling rate from model: ", sampling_rate)
    
    try:
        with AudioSteam(sampling_rate, 10000, AudioStreamType.Input) as mic, AudioSteam(sampling_rate, 9984, AudioStreamType.Output) as speaker, torch.no_grad():
            from openvoice.mel_processing import spectrogram_torch

            while True:
                data = torch.tensor(mic.read()).float().to(converter.device)
                y = data.unsqueeze(0)
                spec = spectrogram_torch(y, filter_length, sampling_rate, hop_length, win_length, center=False).to(converter.device)
                spec_lengths = torch.LongTensor([spec.size(-1)], device=converter.device)
                audio: np.ndarray = converter.model.voice_conversion(spec, spec_lengths, sid_src=src_tone, sid_tgt=dst_tone, tau=tau)[0][0, 0].data.cpu().float().numpy()
                
                speaker.write(audio.tobytes())
    except KeyboardInterrupt:
        print("> Stopping recording...")
        AudioSteam.instance.terminate() # TODO: should use a audio manager to do this

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_color_tone", type=str, help="Specify source color tone pth.", required=True)
    parser.add_argument("-t", "--target_color_tone", type=str, help="Specify target color tone pth.", required=True)
    parser.add_argument("-m", "--model_dir", type=str, help="Specify the checkpoint folder that contained `config.json` and `checkpoint.pth`.", required=True)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("> Using device: ", device)

    tone_color_converter = ToneColorConverter(f'{args.model_dir}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{args.model_dir}/checkpoint.pth')
    print("> Loaded model from ", args.model_dir)

    src_tone = torch.load(args.source_color_tone, map_location=device)
    print("> Loaded source color tone from ", args.source_color_tone)

    dst_tone = torch.load(args.target_color_tone, map_location=device)
    print("> Loaded target color tone from ", args.target_color_tone)

    main(tone_color_converter, src_tone, dst_tone)