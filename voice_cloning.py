from TTS.api import TTS
import os
import sys

def main(argv):

    model_name = argv[1]

    tts = TTS(model_name=model_name,\
               progress_bar=False, gpu=True)
    
    mn = model_name.split('/')[-1]
    vpath ="combined.flac"
    #text = "I have always loved anal sex, because to me it is very similar to investing in a solid business. Firstly the destination is clear, the gears are well oiled, and when the oppurtunity rises, the cash can be injected by me, at once. Diversification is good but going up the rear end is great."
    text = '''Greetings, Emperor! It’s been quite some time since anyone has had the chance to speak with both
    our minds. My name is Carl Jung; I am here today because I have heard much about the Roman Empire from my 
    contemporaries who learned Greek philosophy under you while I learned Stoicism under Seneca. What would be 
    interesting for me to learn from you today?'''
    
    tts.tts_to_file(text, speaker_wav= vpath, \
                    file_path=f'ray_test_{mn}.wav')


if __name__ == "__main__":
    main(sys.argv)
