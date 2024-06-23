#various imports
from time import sleep
import cv2
import torch
import  torchvision.transforms.v2 as v2
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as Model
from torchvision.models.detection import ssd300_vgg16 as Model_
from Model import ModelCNN as Classifier
from torchvision.transforms import InterpolationMode
from dic_signals import classes

video_path = 'vid.mp4' #path of the video where we'll do detection and recognition
cap = cv2.VideoCapture(video_path)  #opening a windows with the video

#trasformation that the image will undego when opened
transform = v2.Compose([
    v2.ToImage(), #transform to tensor image
    v2.ToDtype(dtype=torch.float32) #convert to dtype
])

#trasformation that the image will under go when opened
pp = v2.Compose(
    [   
        v2.ToImage(), #converting to tensor image
        v2.ToDtype(dtype=torch.float32), #converting to dType
        v2.Resize((48, 48), interpolation=InterpolationMode.NEAREST_EXACT), #resizing image to 48x48
        v2.RandomAutocontrast(p=1.0), #applying contrast   
           
    ]
)

device = ('cuda' if torch.cuda.is_available() else 'cpu') #selecting the device


model = Model_() #declaring model for detection
#declaring the classifier for recognition 
classifier = Classifier(input_channels=3, input_shape=48, hidden_units=96, output_shape=43)
#model = Model()

model = model.eval().to(device) #setting detector model to evaluation mode and passing it to the device
classifier = classifier.eval().to(device) #setting recognition model to evaluation mode and passing it to the device

model.load_state_dict(torch.load('saved_model/ssd300/pesi_ok/model_weights.pth')) #loading weights for the detector

classifier.load_state_dict(torch.load('saved_model/CNNModel/pesi_ok/model_weights.pth'))#loading weights for the recognition
#model.load_state_dict(torch.load('saved_model/FastRCNN/ts4/model_weights.pth'))

#reading each frames of the video, the detection and recognition happens every 16 frames
while cap.isOpened():
    f = 0
    ret, frame = cap.read() #reading the frame
    if not ret:
        break

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
    image_tensor = transform(frame).to(device) #applying the trasformations and sending it to the device

    image_tensor = image_tensor.unsqueeze(0) #doing the unsqueeze of the tensor
   
    with torch.no_grad(): #for the test we disable the gradients calculation
        if f%16 == 0: #if the frame is a a multiple of 16 we do detection on it
            #print(image_tensor.shape)
            predictions = model(image_tensor) #doing the detection on the frame
            #print(predictions)

    
    boxes = predictions[0]['boxes'].cpu().numpy() #getting the boxes from the detection
    labels = predictions[0]['labels'].cpu().numpy() #getting the labels from the detection
    scores = predictions[0]['scores'].cpu().numpy() #getting scores from the detection

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.43:   #checking if score exceed the threshold
            box = list(map(int, box)) 
            signal = frame[box[1]:box[3], box[0]:box[2], :] #getting all the pixels of the sign
            x = classifier( pp(signal).unsqueeze(0).to(device) ).argmax(dim=1) #doing the recognition of the sign
            value = x.cpu().detach().numpy().item() #getting the value recognized
            print(classes[value], score) #printing the name of the sign

            #drawing the box around the sign
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            #putting the name of the sign near the box
            cv2.putText(frame, f'Class: {classes[value]}', (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Object Detection', frame) #showing the frame drawed
    f += 1 #increasing the counter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() #releasing the video
cv2.destroyAllWindows() #destroying all windows