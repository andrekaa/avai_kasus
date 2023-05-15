import os
import cv2
import requests
import mimetypes

from doc_ufcn import models
from doc_ufcn.main import DocUFCN

from asrtoolkit import cer, wer

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# definerer og laster layout-modell utenfor funksjon
model_path, parameters = models.download_model('generic-historical-line')

layout = DocUFCN(len(parameters['classes']), parameters['input_size'], 'cuda')
layout.load(model_path, parameters['mean'], parameters['std'])
    
# definerer og laster gjenkjenningsmodell utenfor funksjon
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
ocr = VisionEncoderDecoderModel.from_pretrained("/mnt/md1/NB-HTR/adhoc/checkpoint-9000")

def bounding_box(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we traverse the collection of points only once, 
    to find the min and max for x and y
    """
    bot_left_x, bot_left_y = float('inf'), float('inf')
    top_right_x, top_right_y = float('-inf'), float('-inf')
    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [(bot_left_x, bot_left_y), (top_right_x, top_right_y)]

def ocr_file(img_path: str) -> str:
    transcript = ''

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detected_polygons, probabilities, mask, overlap = layout.predict(image, raw_output=False, mask_output=False, overlap_output=False)
    
    for polygon in detected_polygons[1]:
        bb = bounding_box(polygon['polygon'])
        img = image[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]]

        pixel_values = processor(img, return_tensors="pt").pixel_values
        generated_ids = ocr.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        transcript += generated_text+'\n'
        # Utkommenter hvis man vil inspisere bilde og/eller gjenkjenning linje for linje
        #print(generated_text)
        #plt.imshow(img)
        #plt.show()
    
    with open("transcripts/"+img_path.split('/')[-1].split('.')[0]+".txt", 'w') as f:
        f.write(transcript.strip())
    
    return transcript.strip()

urnids = open('./127588_urnids.txt').read().split('\n')

for urnid in urnids:
    ocr_file("imgs/"+str(urnid)+".jpg")
