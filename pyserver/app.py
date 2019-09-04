from flask import Flask, request, render_template, session, redirect, json
from tensorflow.keras.models import model_from_json
from mtcnn.mtcnn import MTCNN
import boto3, json, pandas as pd, numpy as np, base64, cv2, os, uuid

app = Flask(__name__)

@app.route('/', methods=("POST", "GET"))
def face_recog():
	width = 500
	height = 400
	result_list = []
	table_list = []

	client = boto3.client(
		'rekognition',
		aws_access_key_id='ASIAQWEQC7K55KHET4IG',
		aws_secret_access_key='IQpcZVrndaDi8sfPxwGusRhpe6UIObIY53IT+dX+',
		aws_session_token='FQoGZXIvYXdzEI7//////////wEaDCAOyLswA/F3KbChHCKMArHKtewGmCvnzMad4xwTvdSGpZHeHdmdglbRAUeTEphTo7ks5724wpvZLacWghJ7ft61qTYxXmGlI+3KYlUUJ4s9dBapZkj1Dp+TKC7U9TqKWQC+z/J4RSA9E6NEM9qhlG9GkDSvOlMadCbvWh2mPALJQBFbKyDyy7iUn0aeuQJGcUdD9KVNU2feksPQqjeG51isGng3PUswnmDAyghcnr854KDOr3u4w9M2bJWBXKUkparoFPP9W572LWvgPZ8pCqU/U92NEjNKz/2JEpDJmix5E6SEN++RXs4nwq3+aOICBvzfIiwP2CTc6j0EMlVrb2bh5efBrA8z8MGItUt4wVl5js8wJYgT0i3SKFYo1qWI6wU='
	)

	encoded_img = request.form['encoded_img']
	decoded_img = base64.b64decode(encoded_img)

	filename = str(uuid.uuid4()) + '.jpg'
	# filename = 'result.jpg'
	with open(filename, 'wb') as f:
	    f.write(decoded_img)
	result_img = cv2.imread(filename)

	response = client.detect_faces(Image={'Bytes': decoded_img}, Attributes=['ALL'])

	for ix_main, face in enumerate(response['FaceDetails']):
	    best = 0
	    ix_best = 0
	    for ix, emotion in enumerate(face['Emotions']):
	        if emotion['Confidence'] > best:
	            best = emotion['Confidence']
	            ix_best = ix
	    
	    res = face['Emotions'][ix_best]
	    table_list.append(res)

	    x = int(face['BoundingBox']['Left'] * width)
	    y = int(face['BoundingBox']['Top'] * height)
	    w = int(face['BoundingBox']['Width'] * width)
	    h = int(face['BoundingBox']['Height'] * height)

	    cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

	    font = cv2.FONT_HERSHEY_SIMPLEX
	    cv2.putText(result_img, '[' + str(ix_main + 1) + '] ' + res['Type'],(x, y), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

	cv2.imwrite(filename, result_img)
	with open(filename, 'rb') as image_file:
		encoded_result = base64.b64encode(image_file.read())
		encoded_result = ''.join([chr(item) for item in encoded_result])

	result_list.append(table_list)
	result_list.append(encoded_result)
	
	try: 
		os.remove(filename)
	except: pass

	return json.dumps(result_list)
    # return render_template('simple.html',  tables=[df.to_html(classes='data')], titles='Judulnya Ini')

@app.route('/mtcnn', methods=("POST", "GET"))
def face_recod_mtcnn():
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights('model.h5')

	EMOTIONS = ["Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised", "Neutral"]
	detector = MTCNN()

	result_list = []
	table_list = []

	encoded_img = request.form['encoded_img']
	decoded_img = base64.b64decode(encoded_img)
	
	filename = str(uuid.uuid4()) + '.jpg'
	with open(filename, 'wb') as f:
		f.write(decoded_img)
	result_img = cv2.imread(filename)

	# atur ulang posisi kanal
	result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

	# konversi ke grayscale
	gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

	# deteksi wajah
	faces = detector.detect_faces(result_img)

	# perulangan untuk setiap wajah yang ditemukan
	for ix, result in enumerate(faces):
		# gambar kotak
		x, y, w, h = result['box']
		cv2.rectangle(result_img, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_color = result_img[y:y+h, x:x+w]
		roi_gray = gray[y:y+h, x:x+w]
		if roi_gray.size != 0:
			resized = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
			resized = resized.reshape(1, 48, 48, 1)
			resized = resized.astype('float') / 255.0
			preds = loaded_model.predict(resized)[0]

			emotion_probability = np.max(preds)
			output = EMOTIONS[preds.argmax()]

			res = {'Type': output, 'Confidence': str(emotion_probability * 100)}
			table_list.append(res)

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(result_img, '[' + str(ix + 1) + '] ' + output,(x, y), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

	result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
	cv2.imwrite(filename, result_img)
	with open(filename, 'rb') as image_file:
	    encoded_result = base64.b64encode(image_file.read())
	    encoded_result = ''.join([chr(item) for item in encoded_result])

	result_list.append(table_list)
	result_list.append(encoded_result)
	
	try: 
		os.remove(filename)
	except: pass

	return json.dumps(result_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0')