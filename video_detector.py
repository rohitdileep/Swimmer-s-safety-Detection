import cv2
import numpy as np
import time
import winsound

class ObjectDetection:
    def __init__(self):
        self.MODEL=cv2.dnn.readNet( 'yolov3_final.weights',
            'yolov3.cfg'
        )
        self.CLASSES = []
        with open("coco.names", "r") as f:
            self.CLASSES = [i.strip()  for i in f.readlines()]
        self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i[0] - 1] for i in self.MODEL.getUnconnectedOutLayers()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def detectObj(self , snap):
        height , width , _ = snap.shape
        blob = cv2.dnn.blobFromImage(snap , 1/255, (416,416) , swapRB=True, crop=False)
        self.MODEL.setInput(blob)
        outs = self.MODEL.forward(self.OUTPUT_LAYERS)

        class_ids  =[]
        confidences=[]
        boxes=[]

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y =int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y , w, h = boxes[i]
                labels = str(self.CLASSES[class_ids[i]])
                color = self.COLORS[i]
                cv2.rectangle(snap,(x,y), (x+w , y+h) ,color, 2)
                cv2.putText(snap,labels,(x , y-5) , font , 2, color , 2 )
                if labels == 'shark':
                    winsound.Beep(5000,2000)
        return snap


class VideoStreaming(ObjectDetection):
    def __init__(self):
        super().__init__()
        self.VIDEO = cv2.VideoCapture(0)
        self.Model = ObjectDetection()


    def show(self):
        while(self.VIDEO.isOpened()):
            ret , snap = self.VIDEO.read()
            if ret == True:
                snap = super().detectObj(snap)
                frame = cv2.imencode('.jpg', snap)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.01)
            else:
                break





#










