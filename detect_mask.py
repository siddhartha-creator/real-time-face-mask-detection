import cv2
import numpy as np
from threading import Thread
from tensorflow.keras.models import load_model

IMG_SIZE = 224
class_labels = ['incorrect_mask', 'with_mask', 'without_mask']

# Load trained model
model = load_model("mask_detector_finetuned_3class.keras")

# Load face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


if __name__ == "__main__":

    vs = VideoStream().start()

    while True:
        frame = vs.read()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)) / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)[0]
                class_index = np.argmax(preds)
                label = class_labels[class_index]

                if label == 'with_mask':
                    color = (0, 255, 0)
                elif label == 'without_mask':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Real-Time Mask Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()