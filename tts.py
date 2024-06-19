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

