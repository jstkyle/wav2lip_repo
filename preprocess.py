import json
import subprocess
from pathlib import Path
import platform

import numpy as np
import cv2
from tqdm import tqdm
import torch

import face_detection


device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
PADDING = [0, 10, 0, 0]


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, pads, no_smooth=False):

    batch_size = 1

    predictions = []
    images_array = [cv2.imread(str(image)) for image in images]
    for i in tqdm(range(0, len(images_array), batch_size)):
        predictions.extend(detector.get_detections_for_batch(np.array(images_array[i:i + batch_size])))

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image_array in zip(predictions, images_array):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image_array)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image_array.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image_array.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    bbox_format = "(y1, y2, x1, x2)"
    if not no_smooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    outputs = {
        'bbox': {str(image_path): tuple(map(int, (y1, y2, x1, x2))) for image_path, (x1, y1, x2, y2) in zip(images, boxes)},
        'format': bbox_format
    }
    return outputs


def save_video_frame(video_path, result_dir=None):
    video_path = Path(video_path)
    result_dir = result_dir if result_dir is not None else video_path.with_suffix('')
    result_dir.mkdir(exist_ok=True)
    frame_convert_command = f"ffmpeg -y -i {video_path} -r 25 -f image2 {result_dir}/%05d.jpg"
    return subprocess.call(frame_convert_command, shell=platform.system() != 'Windows')


def save_audio_file(video_path, audio_path=None):
    video_path = Path(video_path)
    audio_path = audio_path if audio_path is not None else video_path.with_suffix('.wav')
    audio_convert_command = f"ffmpeg -y -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 2 {audio_path}"
    subprocess.call(audio_convert_command, shell=platform.system() != 'Windows')


def save_bbox_file(video_path, bbox_dict, json_path=None):
    video_path = Path(video_path)
    json_path = json_path if json_path is not None else video_path.with_suffix('.json')

    with open(json_path, 'w') as f:
        json.dump(bbox_dict, f, indent=4)


if __name__ == '__main__':
    # Argument parsing

    # video_path = "sample/1673_orig.mp4"
    video_path_list = Path("sample").glob("*.mp4")
    for video_path in video_path_list:
        result_dir = Path(video_path).with_suffix('')

        # Split video into image frames with 25 fps
        save_video_frame(video_path=video_path)
        save_audio_file(video_path=video_path)  # bonus

        # Load images, extract bboxes and save the coords(to directly use as array indicies)
        results = face_detect(sorted(list(result_dir.glob("*.jpg"))), pads=PADDING)
        print(video_path)
        print(results)
        print(results['format'])

        save_bbox_file(video_path, results)
