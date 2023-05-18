from flask import Flask, request, Response
from flask_cors import CORS, cross_origin

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jsonpickle
import numpy as np
import cv2
import cv2.dnn
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import ReturnDocument
import json
import base64
import urllib.request
import time
import pandas as pd


CLASSES={0:"carrot",1:"cucumber",2:"potato",3:"lemon",4:"tomato",5:"orange",6:"garlic",7:"onion",8:"lady finger"}

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def countOccurrence(a):
  k = {}
  for j in a:
    if j in k:
      k[j] +=1
    else:
      k[j] =1
  return k

# Initialize the Flask application

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def savetodb(name,type,items):


    uri = "mongodb+srv://lakshyapkhandelwal223:alohomora@cluster0.sgovayn.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
    try:
        db=client.Khana
        inventory=db.inventories
        # inventory.insert_one(x)
        return inventory.find_one_and_update({'name': name},
                        { '$set': { "items" : items,"type":type} },
                        upsert=True,return_document = ReturnDocument.AFTER)
       
    except Exception as e:
        print(e)


def resize_image(img, size=(640,640)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w: 
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)



def main(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = input_image
    original_image=resize_image(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = CLASSES[class_ids[index]]
        detections.append(detection)



    return countOccurrence(detections)


# route http posts to this method
#@app.route('/api/recipe',methods=['POST'])
# @cross_origin
# def recipe():
#     r=json.loads(request.data)
#     items=r['Items']
#     df=pd.read_csv('test.csv')
#     keys=items.keys()
#     result={}
#     for i in keys:
#         if i in df['og']:
            


@app.route('/api/test', methods=['POST'])
@cross_origin()
def test():

    r = json.loads(request.data)
    name=r['name']
    type=r['type']

    if type=='personal':

        base=r['base64Image']
        decoded_data= base64.b64decode((base))
        img_file  = open('image.jpeg','wb')
        img_file.write(decoded_data)
        img_file.close()

    else:
        # capture()
        camport =0
        cam = cv2.VideoCapture(camport)
        result,image=cam.read()
        cv2.imwrite('image.jpeg',image)

       
    img1 = open('image.jpeg','rb').read()

    # convert string of image data to uint8
    nparr = np.frombuffer(img1, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # do some fancy processing here....
    items= main('onnx.onnx',img)
    

    # response={'name':user,'type':type,'items':items}
    print("\n\n items : ",items,"\n\n")
    response = savetodb(name,type,items)
    print("inserted")
    print("\n\n items : ",items,"\n\n")
    
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    print(response_pickled)
    return Response(response=response_pickled, status=200, mimetype="application/json")

def capture():
    url = "http://172.20.10.3/capture?_cb=168245518464"
    filename = "image.jpeg"
    body = request.data
   
    try:
        urllib.request.urlretrieve(url, filename)
        print('Image captured successfully ')
        return 1
    except:
        return 0


# Function to find the matching rows
def find_matching_rows(keywords,df):
    keywords_list = keywords.split(',')
    df['actual ingredients'].fillna('', inplace=True)
    vectorizer = CountVectorizer()
    keyword_strings = df['actual ingredients'].tolist()
    keyword_vectors = vectorizer.fit_transform(keyword_strings)
    input_vector = vectorizer.transform([keywords])
    similarities = cosine_similarity(input_vector, keyword_vectors)
    df['similarity'] = similarities[0]
    df_selected = df.loc[:,  ['RecipeID', 'recipe_name', 'Ingredients', 'Directions', 'similarity']]
    df_sorted = df_selected.sort_values(by='similarity', ascending=False)
    top_5_rows = df_sorted.head(5)
    top_5_rows = top_5_rows.reset_index(drop=True)
    top_5_dict = top_5_rows.to_dict(orient='records')
    return top_5_dict
    

@app.route('/api/getRecipe', methods=['POST'])
@cross_origin()
def getRecipe():
    r = json.loads(request.data)
    name=r['name']
    type=r['type']
    items=r['items']

    # Convert the JSON object to a Python dictionary
   
    keysList = list(items.keys())
    my_string = ", ".join(keysList)
    

    # Read CSV file into a pandas dataframe
    df = pd.read_csv('recipes.csv')
    # Test the function with sample input and output
    keywords = my_string
    result = find_matching_rows(keywords,df)


    # Convert the result to a JSON string
    result_json = json.dumps(result, indent=4)
    
    return result_json








    

app.run(host="0.0.0.0", port=5000,debug=True)