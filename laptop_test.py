#!/usr/bin/python3
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

model_path="model.tflite"
p = 0
beta = 0.15

interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

vid = cv2.VideoCapture(0)

while True:
	ret,frame = vid.read()
	input_data = np.array([cv2.resize(frame,(320,320))],dtype=np.ubyte)
	interpreter.set_tensor(input_details[0]['index'], input_data)
	interpreter.invoke()
	classes = interpreter.get_tensor(output_details[1]['index'])
	probablities = interpreter.get_tensor(output_details[2]['index'])
	if classes[0][0] == 0.0 and probablities[0][0] >= 0.5:
		p = beta*p + (1-beta)*1
	else:
		p = beta*p + (1-beta)*0
	if p >= 0.5:
		print('on')
	else:
		print('off')

cap.release()
