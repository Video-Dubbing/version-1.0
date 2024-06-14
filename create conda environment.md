# Create and activate conda environment
conda create -y -n translatube python=3.11
conda activate translatube

<!-- # Install whisper (speech to text)
pip install git+https://github.com/openai/whisper.git
sudo apt update && sudo apt install ffmpeg

# Install bark (text to speech)
pip install git+https://github.com/suno-ai/bark.git -->

# Install seamlessM4t (translate)
pip install fairseq2
pip install git+https://github.com/facebookresearch/seamless_communication
pip install gradio
pip install huggingface-hub
pip install torch
pip install torchaudio
<!-- pip install pysndfile==1.0.0 -->
mamba install -y -c conda-forge libsndfile==1.0.31 pyperclip ipywidgets

# Download videos
pip install twitch-dl
pip install pytube