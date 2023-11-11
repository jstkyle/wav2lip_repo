from pathlib import Path
import time
import argparse
import json
import subprocess
import platform

import numpy as np
import cv2
from tqdm import tqdm
import torch

from models import Wav2Lip, Wav2Lip_noRes
import audio

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print('Using {} for inference.'.format(device))

BATCH_SIZE = 1
WAV2LIP_BATCH_SIZE = 1
IMG_SIZE = 96
VIDEO_FPS = 25
FRAME_H = 224
FRAME_W = 224
MEL_STEP_SIZE = 16
SAMPLING_RATE = 16000
ORIGINAL_CHECKPOINT_PATH = "checkpoints/lrs3_e16a32d32.pth"
COMPRESSED_CHECKPOINT_PATH = "checkpoints/lrs3_e4a8d8_noRes.pth"


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model_nota(cls, path, **kwargs):
    model = cls(**kwargs)
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    return model.eval()

def count_params(model):
    return sum(p.numel() for p in model.parameters())
    

class VideoSlicer:
    def __init__(self, args, frame_dir, bbox_path, video_path=None):
        self.args = args
        self.fps = VIDEO_FPS
        self.frame_dir = frame_dir
        self.frame_path_list = sorted(list(Path(self.frame_dir).glob("*.jpg")))
        self.frame_array_list = [cv2.imread(str(image)) for image in self.frame_path_list]

        with open(bbox_path, 'r') as f:
            metadata = json.load(f)
            self.bbox = [metadata['bbox'][key] for key in sorted(metadata['bbox'].keys())]
            self.bbox_format = metadata['format']
        assert len(self.bbox) == len(self.frame_array_list)
        self._video_path = video_path

    @property
    def video_path(self):
        return self._video_path

    def __len__(self):
        return len(self.frame_array_list)

    def __getitem__(self, idx):
        bbox = self.bbox[idx]
        frame_original = self.frame_array_list[idx]
        # return frame_original[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        return frame_original, bbox


class AudioSlicer:
    def __init__(self, args, audio_path):
        self.args = args
        self.fps = VIDEO_FPS
        self.mel_chunks = self._audio_chunk_generator(audio_path)
        self._audio_path = audio_path

    @property
    def audio_path(self):
        return self._audio_path
    
    def __len__(self):
        return len(self.mel_chunks)

    def _audio_chunk_generator(self, audio_path):
        wav = audio.load_wav(audio_path, SAMPLING_RATE)
        mel = audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / self.fps

        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + MEL_STEP_SIZE > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - MEL_STEP_SIZE:])
                return mel_chunks
            mel_chunks.append(mel[:, start_idx: start_idx + MEL_STEP_SIZE])
            i += 1

    def __getitem__(self, idx):
        return self.mel_chunks[idx]


class Wav2LipCompressionDemo:
    def __init__(self, args, result_dir='./temp') -> None:
        self.args = args
        self.video_dict = {}
        self.audio_dict = {}
        self.model_original = load_model_nota(Wav2Lip, ORIGINAL_CHECKPOINT_PATH)
        self.model_compressed = load_model_nota(Wav2Lip_noRes, COMPRESSED_CHECKPOINT_PATH, nef=4, naf=8, ndf=8)

        self.params_original = f"{(count_params(self.model_original)/1e6):.1f}M"
        self.params_compressed = f"{(count_params(self.model_compressed)/1e6):.1f}M"

        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)
    
    def update_audio(self, audio_path, name=None):
        _name = name if name is not None else Path(audio_path).stem
        self.audio_dict.update(
            {_name: AudioSlicer(self.args, audio_path)}
        )

    def update_video(self, frame_dir_path, bbox_path, video_path=None, name=None):
        _name = name if name is not None else Path(frame_dir_path).stem
        self.video_dict.update(
            {_name: VideoSlicer(self.args, frame_dir_path, bbox_path, video_path=video_path)}
        )

    @staticmethod
    def _paired_data_iterator(audio_iterable, video_iterable):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        for i, m in enumerate(audio_iterable):
            idx = i % len(video_iterable)
            _frame_to_save, coords = video_iterable[idx]
            frame_to_save = _frame_to_save.copy()
            face = frame_to_save[coords[0]:coords[1], coords[2]:coords[3]].copy()

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= WAV2LIP_BATCH_SIZE:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, IMG_SIZE//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, IMG_SIZE//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _infer(self, audio_name, video_name, model_type='original'):
        audio_iterable = self.audio_dict[audio_name]
        video_iterable = self.video_dict[video_name]
        data_iterator = self._paired_data_iterator(audio_iterable, video_iterable)

        for (img_batch, mel_batch, frames, coords) in tqdm(data_iterator,
                                                           total=int(np.ceil(float(len(audio_iterable)) / WAV2LIP_BATCH_SIZE))):

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            with torch.no_grad():
                if model_type == 'original':
                    preds = self.model_original(mel_batch, img_batch)
                elif model_type == 'compressed':
                    preds = self.model_compressed(mel_batch, img_batch)
                else:
                    raise ValueError(f"`model_type` should be either `original` or `compressed`!")

            preds = preds.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            for pred, frame, coord in zip(preds, frames, coords):
                y1, y2, x1, x2 = coord
                pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))

                frame[y1:y2, x1:x2] = pred
                yield frame

    def save_as_video(self, audio_name, video_name, model_type):

        output_video_path = self.result_dir / 'original_voice.mp4'
        frame_only_video_path = self.result_dir / 'original.mp4'
        audio_path = self.audio_dict[audio_name].audio_path

        out = cv2.VideoWriter(str(frame_only_video_path),
                              cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (FRAME_W, FRAME_H))
        start = time.time()
        for frame in self._infer(audio_name=audio_name, video_name=video_name, model_type=model_type):
            out.write(frame)
        inference_time = time.time() - start
        out.release()

        command = f"ffmpeg -hide_banner -loglevel error -y -i {audio_path} -i {frame_only_video_path} -strict -2 -q:v 1 {output_video_path}"
        subprocess.call(command, shell=platform.system() != 'Windows')

        # The number of frames of generated video
        video_frames_num = len(self.audio_dict[audio_name])
        inference_fps = video_frames_num / inference_time

        return output_video_path, inference_time, inference_fps


def get_parsed_args():

    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    args = parser.parse_args()

    return args


def main():
    args = get_parsed_args()

    demo_generator = Wav2LipCompressionDemo(args)
    demo_generator.update_audio("sample/1673_orig.wav", name="1673")
    demo_generator.update_audio("sample/4598_orig.wav", name="4598")
    demo_generator.update_video("sample/2145_orig", "sample/2145_orig.json", name="2145")
    demo_generator.update_video("sample/2942_orig", "sample/2942_orig.json", name="2942")

    processed_time = []
    for i in range(5):
        start = time.time()
        out = cv2.VideoWriter('temp/original.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (FRAME_W, FRAME_H))
        for frame in demo_generator.infer(audio_name="4598", video_name="2145", model_type="original"):
            out.write(frame)
        out.release()
        processed_time.append(time.time() - start)

    command = f"ffmpeg -hide_banner -loglevel error -y -i {'sample/4598_orig.wav'} -i {'temp/original.mp4'} -strict -2 -q:v 1 {'temp/original_voice.mp4'}"
    subprocess.call(command, shell=platform.system() != 'Windows')
    print(f"Processed time: {np.mean(processed_time)}")

    processed_time = []
    for i in range(5):
        start = time.time()
        out = cv2.VideoWriter('temp/compressed.mp4',
                              cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (FRAME_W, FRAME_H))
        for frame in demo_generator.infer(audio_name="4598", video_name="2145", model_type="compressed"):
            out.write(frame)
        out.release()
        processed_time.append(time.time() - start)

    command = f"ffmpeg -hide_banner -loglevel error -y -i {'sample/4598_orig.wav'} -i {'temp/compressed.mp4'} -strict -2 -q:v 1 {'temp/compressed_voice.mp4'}"
    subprocess.call(command, shell=platform.system() != 'Windows')
    print(f"Processed time: {np.mean(processed_time)}")


if __name__ == '__main__':
    main()
