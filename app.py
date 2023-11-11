import os
import subprocess
import threading
from pathlib import Path

import gradio as gr

from demo import Wav2LipCompressionDemo

LRS_ORIGINAL_URL = os.environ['LRS_ORIGINAL_URL']
LRS_COMPRESSED_URL = os.environ['LRS_COMPRESSED_URL']
subprocess.call(f"wget --no-check-certificate -O {'checkpoints/lrs3_e16a32d32.pth'} {LRS_ORIGINAL_URL}", shell=True)
subprocess.call(f"wget --no-check-certificate -O {'checkpoints/lrs3_e4a8d8_noRes.pth'} {LRS_COMPRESSED_URL}", shell=True)

lock = threading.Lock()  # lock for asserting that concurrency_count == 1

VIDEO_LABEL_LIST = ['v1', 'v2', 'v3', 'v4', 'v5']
AUDIO_LABEL_LIST = ['a1', 'a2', 'a3', 'a4', 'a5']

class Wav2LipCompressionGradio(Wav2LipCompressionDemo):

    @staticmethod
    def _is_valid_input(video_selection, audio_selection):
        assert video_selection in VIDEO_LABEL_LIST, f"Your input ({video_selection}) is not in {VIDEO_LABEL_LIST}!!!"
        assert audio_selection in AUDIO_LABEL_LIST, f"Your input ({audio_selection}) is not in {AUDIO_LABEL_LIST}!!!"

    def generate_original_model(self, video_selection, audio_selection):
        try:
            self._is_valid_input(video_selection, audio_selection)

            with lock:
                output_video_path, inference_time, inference_fps = \
                    self.save_as_video(audio_name=audio_selection,
                                       video_name=video_selection,
                                       model_type="original")

                return str(output_video_path), format(inference_time, ".2f"), format(inference_fps, ".1f")
        except KeyboardInterrupt as e:
            exit()
        except Exception as e:
            print(e)
            pass

    def generate_compressed_model(self, video_selection, audio_selection):
        try:
            self._is_valid_input(video_selection, audio_selection)

            with lock:
                output_video_path, inference_time, inference_fps = \
                    self.save_as_video(audio_name=audio_selection,
                                       video_name=video_selection,
                                       model_type="compressed")

                return str(output_video_path), format(inference_time, ".2f"), format(inference_fps, ".1f")
        except KeyboardInterrupt as e:
            exit()
        except Exception as e:
            print(e)
            pass

    def switch_video_samples(self, video_selection):
        try:
            if video_selection == VIDEO_LABEL_LIST[0]:
                sample_video_path = "sample/2145_orig.mp4"
            elif video_selection == VIDEO_LABEL_LIST[1]:
                sample_video_path = "sample/2942_orig.mp4"
            elif video_selection == VIDEO_LABEL_LIST[2]:
                sample_video_path = "sample/4598_orig.mp4"
            elif video_selection == VIDEO_LABEL_LIST[3]:
                sample_video_path = "sample/4653_orig.mp4"
            elif video_selection == VIDEO_LABEL_LIST[4]:
                sample_video_path = "sample/13692_orig.mp4"
            else:   # default sample
                sample_video_path = "sample/2145_orig.mp4"
            return sample_video_path

        except KeyboardInterrupt as e:
            exit()
        except Exception as e:
            print(e)
            pass

    def switch_audio_samples(self, audio_selection):
        try:
            if audio_selection == AUDIO_LABEL_LIST[0]:
                sample_audio_path = "sample/1673_orig.wav"
            elif audio_selection == AUDIO_LABEL_LIST[1]:
                sample_audio_path = "sample/9948_orig.wav"
            elif audio_selection == AUDIO_LABEL_LIST[2]:
                sample_audio_path = "sample/11028_orig.wav"
            elif audio_selection == AUDIO_LABEL_LIST[3]:
                sample_audio_path = "sample/12640_orig.wav"
            elif audio_selection == AUDIO_LABEL_LIST[4]:
                sample_audio_path = "sample/5592_orig.wav"
            else:  # default sample
                sample_audio_path = "sample/1673_orig.wav"
            return sample_audio_path

        except KeyboardInterrupt as e:
            exit()
        except Exception as e:
            print(e)
            pass



if __name__ == "__main__":

    servicer = Wav2LipCompressionGradio(args=None)
    servicer.update_video("sample/2145_orig", "sample/2145_orig.json",
                          video_path="sample/2145_orig.mp4",
                          name=VIDEO_LABEL_LIST[0])
    servicer.update_video("sample/2942_orig", "sample/2942_orig.json",
                          video_path="sample/2942_orig.mp4",
                          name=VIDEO_LABEL_LIST[1])
    servicer.update_video("sample/4598_orig", "sample/4598_orig.json",
                          video_path="sample/4598_orig.mp4",
                          name=VIDEO_LABEL_LIST[2])
    servicer.update_video("sample/4653_orig", "sample/4653_orig.json",
                          video_path="sample/4653_orig.mp4",
                          name=VIDEO_LABEL_LIST[3])
    servicer.update_video("sample/13692_orig", "sample/13692_orig.json",
                          video_path="sample/13692_orig.mp4",
                          name=VIDEO_LABEL_LIST[4])         
    servicer.update_audio("sample/1673_orig.wav", name=AUDIO_LABEL_LIST[0])
    servicer.update_audio("sample/9948_orig.wav", name=AUDIO_LABEL_LIST[1])
    servicer.update_audio("sample/11028_orig.wav", name=AUDIO_LABEL_LIST[2])
    servicer.update_audio("sample/12640_orig.wav", name=AUDIO_LABEL_LIST[3])
    servicer.update_audio("sample/5592_orig.wav", name=AUDIO_LABEL_LIST[4])

    with gr.Blocks(theme='nota-ai/theme') as demo:
        gr.Markdown(Path('docs/header.md').read_text())
        gr.Markdown(Path('docs/description.md').read_text())
        with gr.Row():
            with gr.Column(variant='panel'):

                gr.Markdown('<h2 align="center">Select input video and audio</h2>')
                # Define samples
                sample_video = gr.Video(type="mp4", label="Input Video")
                sample_audio = gr.Audio(interactive=False, label="Input Audio")

                # Define radio inputs
                video_selection = gr.components.Radio(VIDEO_LABEL_LIST,
                                                      type='value', label="Select an input video:")
                audio_selection = gr.components.Radio(AUDIO_LABEL_LIST,
                                                      type='value', label="Select an input audio:")
                # Define button inputs
                with gr.Row().style(equal_height=True):
                    generate_original_button = gr.Button(value="Generate with Original Model", variant="primary")
                    generate_compressed_button = gr.Button(value="Generate with Compressed Model", variant="primary")
            with gr.Column(variant='panel'):
                # Define original model output components
                gr.Markdown('<h2 align="center">Original Wav2Lip</h2>')
                original_model_output = gr.Video(type="mp4", label="Original Model")
                with gr.Column():
                    with gr.Row().style(equal_height=True):
                        original_model_inference_time = gr.Textbox(value="", label="Total inference time (sec)")
                        original_model_fps = gr.Textbox(value="", label="FPS")
                    original_model_params = gr.Textbox(value=servicer.params_original, label="# Parameters")
            with gr.Column(variant='panel'):
                # Define compressed model output components
                gr.Markdown('<h2 align="center">Compressed Wav2Lip (Ours)</h2>')
                compressed_model_output = gr.Video(type="mp4", label="Compressed Model")
                with gr.Column():
                    with gr.Row().style(equal_height=True):
                        compressed_model_inference_time = gr.Textbox(value="", label="Total inference time (sec)")
                        compressed_model_fps = gr.Textbox(value="", label="FPS")
                    compressed_model_params = gr.Textbox(value=servicer.params_compressed, label="# Parameters")

        # Switch video and audio samples when selecting the raido button
        video_selection.change(fn=servicer.switch_video_samples, inputs=video_selection, outputs=sample_video)
        audio_selection.change(fn=servicer.switch_audio_samples, inputs=audio_selection, outputs=sample_audio)

        # Click the generate button for original model
        generate_original_button.click(servicer.generate_original_model,
                                       inputs=[video_selection, audio_selection],
                                       outputs=[original_model_output, original_model_inference_time, original_model_fps])
        # Click the generate button for compressed model
        generate_compressed_button.click(servicer.generate_compressed_model,
                                         inputs=[video_selection, audio_selection],
                                         outputs=[compressed_model_output, compressed_model_inference_time, compressed_model_fps])

        gr.Markdown(Path('docs/footer.md').read_text())

    demo.queue(concurrency_count=1)
    demo.launch()
