import gradio as gr
import pyperclip
import urllib.parse as urlparse
from pytube import YouTube
import re
import subprocess
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, pipeline
from lang_list import union_language_dict
import torch
import whisper
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import gc
from tts import text_to_speech

# get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Create a dictionary to store the language codes
language_dict = union_language_dict()

YOUTUBE = "youtube"
TWITCH = "twitch"
ERROR = "error"

def copy_url_from_clipboard():
    return pyperclip.paste()

def clear_video_url():
    visible = False
    image = gr.Image(visible=visible, scale=1)
    source_languaje = gr.Dropdown(visible=visible, label="Source languaje", show_label=True, value="English", choices=language_dict, scale=1, interactive=True)
    target_languaje = gr.Dropdown(visible=visible, label="Target languaje", show_label=True, value="Español", choices=language_dict, scale=1, interactive=True)
    translate_button = gr.Button(size="lg", value="translate", min_width="10px", scale=0, visible=visible)
    original_audio = gr.Audio(label="Original audio", elem_id="original_audio", visible=visible, interactive=False)
    original_audio_transcribed = gr.Textbox(label="Original audio transcribed", elem_id="original_audio_transcribed", interactive=False, visible=visible)
    original_audio_translated = gr.Textbox(label="Original audio translated", elem_id="original_audio_translated", interactive=False, visible=visible)
    translated_audio = gr.Audio(label="Translated audio", elem_id="translated_audio", visible=visible)
    return (
        "",
        image, 
        source_languaje, 
        target_languaje, 
        translate_button, 
        original_audio, 
        original_audio_transcribed, 
        translated_audio, 
        original_audio_translated,
    )

def get_youtube_thumbnail(video_id):
    thumbnail_url = f"https://img.youtube.com/vi/{video_id}/0.jpg"
    return thumbnail_url

def get_youtube_video_id(url):
    if "youtu.be" in url.lower():
        yt = YouTube(url)
        thumbnail_url = yt.thumbnail_url
        return thumbnail_url
    else:
        parsed_url = urlparse.urlparse(url)
        video_id = urlparse.parse_qs(parsed_url.query).get('v')
        if video_id:
            thumbnail_url = get_youtube_thumbnail(video_id[0])
            return thumbnail_url
        else:
            return None

def is_valid_url(url):
    source_languaje = gr.Dropdown(visible=True, label="Source languaje", show_label=True, value="English", choices=language_dict, scale=1, interactive=True)
    target_languaje = gr.Dropdown(visible=True, label="Target languaje", show_label=True, value="Español", choices=language_dict, scale=1, interactive=True)
    translate_button = gr.Button(size="lg", value="translate", min_width="10px", scale=0, visible=True)
    original_audio = gr.Audio(label="Original audio", elem_id="original_audio", visible=True, interactive=False)
    original_audio_transcribed = gr.Textbox(label="Original audio transcribed", elem_id="original_audio_transcribed", interactive=False, visible=True)
    original_audio_translated = gr.Textbox(label="Original audio translated", elem_id="original_audio_translated", interactive=False, visible=True)
    translated_audio = gr.Audio(label="Translated audio", elem_id="translated_audio", visible=True)
    if "youtube" in url.lower() or "youtu.be" in url.lower():
        thumbnail = get_youtube_video_id(url)
        if thumbnail:
            return (
                gr.Image(value=thumbnail, visible=True, show_download_button=False, container=False), 
                source_languaje,
                target_languaje,
                translate_button, 
                gr.Textbox(value=YOUTUBE, label="Stream page", elem_id="stream_page", visible=False),
                original_audio,
                original_audio_transcribed, 
                translated_audio,
                original_audio_translated, 
            )
        else:
            return (
                gr.Image(value="assets/youtube-no-thumbnails.webp", visible=True, show_download_button=False, container=False), 
                source_languaje,
                target_languaje,
                translate_button, 
                gr.Textbox(value=YOUTUBE, label="Stream page", elem_id="stream_page", visible=False),
                original_audio,
                original_audio_transcribed, 
                translated_audio,
                original_audio_translated, 
            )
    elif "twitch" in url.lower() or "twitch.tv" in url.lower():
        return (
            gr.Image(value="assets/twitch.webp", visible=True, show_download_button=False, container=False), 
            source_languaje,
            target_languaje,
            translate_button, 
            gr.Textbox(value=TWITCH, label="Stream page", elem_id="stream_page", visible=False),
            original_audio,
            original_audio_transcribed, 
            translated_audio,
            original_audio_translated, 
        )
    else:
        visible = False
        image = gr.Image(value="assets/youtube_error.webp", visible=visible, show_download_button=False, container=False)
        source_languaje = gr.Dropdown(visible=visible, label="Source languaje", show_label=True, value="English", choices=language_dict, scale=1, interactive=True)
        target_languaje = gr.Dropdown(visible=visible, label="Target languaje", show_label=True, value="Español", choices=language_dict, scale=1, interactive=True)
        translate_button = gr.Button(size="lg", value="translate", min_width="10px", scale=0, visible=visible)
        stream_page = gr.Textbox(value=ERROR, label="Stream page", elem_id="stream_page", visible=visible)
        original_audio = gr.Audio(label="Original audio", elem_id="original_audio", visible=visible, interactive=False)
        original_audio_transcribed = gr.Textbox(label="Original audio transcribed", elem_id="original_audio_transcribed", interactive=False, visible=visible)
        original_audio_translated = gr.Textbox(label="Original audio translated", elem_id="original_audio_translated", interactive=False, visible=visible)
        translated_audio = gr.Audio(label="Translated audio", elem_id="translated_audio", visible=visible)
        return (
            image, 
            source_languaje,
            target_languaje,
            translate_button, 
            stream_page,
            original_audio,
            original_audio_transcribed, 
            translated_audio,
            original_audio_translated, 
        )

def get_audio_from_video(url, stream_page):
    if stream_page == YOUTUBE:
        yt = YouTube(url)
        audio_streams = yt.streams.filter(mime_type="audio/mp4")

        # Get all available audio bitrates
        abr_list = []
        for stream in audio_streams:
            abr_list.append(stream.abr)
        abr_list = sorted(set(abr_list))

        # Get the highest audio bitrate
        audio_stream = audio_streams.filter(abr=abr_list[0]).first()

        # Download the audio
        filename = "audio.mp3"
        audio_stream.download(filename=filename)

        return (
            gr.Audio(value=filename, label="Original audio", elem_id="original_audio", visible=True, interactive=False),
            gr.Textbox(value=filename, label="Stream page", elem_id="stream_page", visible=False)
        )
    elif stream_page == TWITCH:
        # Get the video id
        video_id = re.search("\d{10}", url).group(0)

        # Download the video
        filename = "audio.mkv"
        subprocess.run(["twitch-dl", "download", "--overwrite", "-q", "audio_only", "--output", filename, video_id])

        return (
            gr.Audio(value=filename, label="Original audio", elem_id="original_audio", visible=True, interactive=False),
            gr.Textbox(value=filename, label="Stream page", elem_id="stream_page", visible=False)
        )

def trascribe_audio(audio_path, source_lang):
    # Load the model
    trascribe_model = whisper.load_model("large-v2", device=device)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(trascribe_model.device)
    
    # Decode the result
    options = whisper.DecodingOptions(fp16 = False, language = language_dict[source_lang]['transcriber'])
    result = whisper.decode(trascribe_model, mel, options)

    # Save the result to a file
    filename = "result.txt"
    with open(filename, "w") as f:
        f.write(result.text)
    
    # Remove audio file
    subprocess.run(["rm", audio_path])

    # free gpu memory
    del trascribe_model
    del audio
    del mel
    del options
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    
    return (
        result.text,
        gr.Textbox(value=filename, label="Original audio transcribed", elem_id="original_audio_transcribed", visible=False)
    )

def translate(original_audio_transcribed_path, source_languaje, target_languaje):
    # model
    translate_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    translate_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    # Get source and target languaje codes
    source_languaje_code = language_dict[source_languaje]["translator"]
    target_languaje_code = language_dict[target_languaje]["translator"]

    # Get the transcribed text
    with open(original_audio_transcribed_path, "r") as f:
        transcribed_text = f.read()
    
    # Translate the text
    encoded = translate_tokenizer(transcribed_text, return_tensors="pt").to(device)
    generated_tokens = translate_model.generate(
        **encoded,
        forced_bos_token_id=translate_tokenizer.lang_code_to_id[target_languaje_code]
    )
    translated = translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # Save the result to a file
    filename = "translated_text.txt"
    with open(filename, "w") as f:
        f.write(translated)
    
    # Remove transcribed file
    subprocess.run(["rm", original_audio_transcribed_path])

    # free gpu memory
    del translate_model
    del translate_tokenizer
    del encoded
    del generated_tokens
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return (
        translated,
        gr.Textbox(value=filename, label="Original audio translated", elem_id="original_audio_translated", visible=False)
    )

translated_audio_path = "translated_audio.wav"
text_to_speech(original_audio_translated_path, output_file_path, language=target_language, slow=slow_speed)

def delete_translated_audio(translated_audio_path):
    subprocess.run(["rm", translated_audio_path])

with gr.Blocks() as demo:
    # Layout
    with gr.Row(variant="panel"):
        url_textbox = gr.Textbox(placeholder="Add video URL here", label="Video URL", elem_id="video_url", scale=1, interactive=True)
        copy_button   = gr.Button(size="sm", icon="icons/copy.svg",   value="", min_width="10px", scale=0)
        delete_button = gr.Button(size="sm", icon="icons/delete.svg", value="", min_width="10px", scale=0)

    stream_page = gr.Textbox(label="Stream page", elem_id="stream_page", visible=False)
    visible = False
    with gr.Row(equal_height=False):
        image = gr.Image(visible=visible, scale=1)
        with gr.Column():
            with gr.Row():
                source_languaje = gr.Dropdown(visible=visible, label="Source languaje", show_label=True, value="English", choices=language_dict, scale=1, interactive=True)
                target_languaje = gr.Dropdown(visible=visible, label="Target languaje", show_label=True, value="Español", choices=language_dict, scale=1, interactive=True)
            with gr.Row():
                translate_button = gr.Button(size="lg", value="translate", min_width="10px", scale=0, visible=visible)

    original_audio = gr.Audio(label="Original audio", elem_id="original_audio", visible=visible, interactive=False)
    original_audio_path = gr.Textbox(label="Stream page", elem_id="stream_page", visible=False)
    original_audio_transcribed = gr.Textbox(label="Original audio transcribed", elem_id="original_audio_transcribed", interactive=False, visible=visible)
    original_audio_transcribed_path = gr.Textbox(label="Original audio transcribed", elem_id="original_audio_transcribed", visible=False)
    original_audio_translated = gr.Textbox(label="Original audio translated", elem_id="original_audio_translated", interactive=False, visible=visible)
    original_audio_translated_path = gr.Textbox(label="Original audio translated", elem_id="original_audio_translated", visible=False)
    translated_audio = gr.Audio(label="Translated audio", elem_id="translated_audio", visible=visible)
    translated_audio_translated_path = gr.Textbox(label="translated audio translated", elem_id="translated_audio_translated", visible=False)

    # Events
    copy_button.click(fn=copy_url_from_clipboard, outputs=url_textbox)
    delete_button.click(
        fn=clear_video_url, 
        outputs=[
            url_textbox, 
            image, 
            source_languaje, 
            target_languaje, 
            translate_button, 
            original_audio, 
            original_audio_transcribed, 
            translated_audio, 
            original_audio_translated,
        ]
    )
    url_textbox.change(
        fn=is_valid_url, 
        inputs=url_textbox, 
        outputs=[
            image, 
            source_languaje, 
            target_languaje, 
            translate_button, 
            stream_page, 
            original_audio, 
            original_audio_transcribed, 
            translated_audio, 
            original_audio_translated,
        ]
    )
    translate_button.click(fn=get_audio_from_video, inputs=[url_textbox, stream_page], outputs=[original_audio, original_audio_path])
    original_audio.change(fn=trascribe_audio, inputs=[original_audio_path, source_languaje], outputs=[original_audio_transcribed, original_audio_transcribed_path])
    original_audio_transcribed.change(fn=translate, inputs=[original_audio_transcribed_path, source_languaje, target_languaje], outputs=[original_audio_translated, original_audio_translated_path])
    original_audio_translated.change(fn=text_to_speech, inputs=original_audio_translated_path, outputs=translated_audio)
    translated_audio.change(fn=delete_translated_audio, inputs=translated_audio)


demo.launch(share=True)
