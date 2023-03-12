from gtts import gTTS
import os
tts = gTTS(text='中文呢', lang='zh')
tts.save("good.mp3")
os.system("mpg321 good.mp3")