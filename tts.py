from gtts import gTTS
import os
def text_to_speech(input_file, output_file, language='te', slow=False):
    try:
        with open(input_file, "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    speech = gTTS(text=text, lang=language, slow=slow)
    speech.save(output_file)
    print(f"Speech saved to {output_file}")

    # Play the generated speech
    os.system(f"start {output_file}")

if __name__ == "__main__":
    input_file_path = "C:/Users/DELL/Desktop/telugu.txt.txt"
    output_file_path = "voice.mp3"
    target_language = 'te'
    slow_speed = False

    text_to_speech(input_file_path, output_file_path, language=target_language, slow=slow_speed)
