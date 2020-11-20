# Emotions of text

from nrclex import NRCLex

text_file = open('sad.txt')
text = text_file.read()

text_object = NRCLex(text)

#print(dir('text_object'))

print(text_object.raw_emotion_scores)
print(text_object.top_emotions)
print(text_object.affect_frequencies)


