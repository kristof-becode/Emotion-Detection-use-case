from moviepy.editor import *

mp4_file = 'WIN_20201113_13_45_57_Pro.mp4'

mp3_file = 'audio.mp3'

videoClip = VideoFileClip(mp4_file)

audioClip = videoClip.audio

audioClip.write_audiofile(mp3_file)

audioClip.close()
videoClip.close()





