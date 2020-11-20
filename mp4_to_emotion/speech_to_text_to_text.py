import ibm_watson
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import re

#copy from ibm website (after making account, and press 'create',
# https://cloud.ibm.com/services/speech-to-text/crn%3Av1%3Abluemix%3Apublic%3Aspeech-to-text%3Aeu-gb%3Aa%2F62b8e5711e364f34bc78c3d204b38455%3A9088022c-c305-4b06-914c-763307174248%3A%3A?paneId=manage&new=true)
apikey = "_qv4s3PJZspxGQLM46EGViROvdbRQlSyxDWkT1sLmxVX"
url = "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/9088022c-c305-4b06-914c-763307174248"

#set up service
authenticator = IAMAuthenticator(apikey)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url)

#open audio source and convert
#with open('happy.mp3', 'rb') as f:
#    res = stt.recognize(audio= f , content_type='audio/mp3', model= 'en-US_NarrowbandModel', continious= True).get_result()
#print(res) #transcript

with open('audio.mp3', 'rb') as f:
    res = stt.recognize(audio= f , content_type='audio/mp3', model= 'en-US_NarrowbandModel', continious= True).get_result()
print(res) #transcript

########################################################################

#write transcript to .txt file

res_string = str(res)
print(res_string[81:-26])

#x = re.findall("^'transcript': '.*'$", res_string)
#print(x)








