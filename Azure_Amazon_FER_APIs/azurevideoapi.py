# To launch app in terminal use command 'streamlit run azurevideoapi.py'

import streamlit as st

# Uses Azure Media Services - videoindexer.ai

#Available options: people, animatedCharacters ,keywords, labels, sentiments, emotions, topics, keyframes, transcript, ocr, speakers, scenes, and namedEntities.

# Load embeded player from videoindexer.ai
st.components.v1.iframe("https://www.videoindexer.ai/embed/player/00000000-0000-0000-0000-000000000000/4dc0aa32bf",width = 560, height = 315)

# Load embeded insights from videoindexer.ai
st.components.v1.iframe("https://www.videoindexer.ai/embed/insights/00000000-0000-0000-0000-000000000000/4dc0aa32bf", width = 680, height= 780)
#<script src="https://breakdown.blob.core.windows.net/public/vb.widgets.mediator.js"></script>
