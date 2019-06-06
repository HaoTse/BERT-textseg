import re

# sentence split token
punkt_token = re.compile(r'[。，；]|[!?！？]+')

def chinese_punkt(text):
    return re.split(punkt_token, text)

def split_sentences(text):
    sentences = chinese_punkt(text)
    ret_sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    return ret_sentences
