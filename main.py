import cv2
import numpy as np

# Load model
net = cv2.dnn.readNetFromTensorflow(
    'models/frozen_inference_graph.pb',
    'models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
)

CLASSES = ["background","person","bicycle","car","motorcycle",
           "airplane","bus","train","truck","boat","traffic light",
           "fire hydrant","stop sign","parking meter","bench","bird",
           "cat","dog","horse","sheep","cow","elephant","bear",
           "zebra","giraffe","backpack","umbrella","handbag","tie",
           "suitcase","frisbee","skis","snowboard","sports ball",
           "kite","baseball bat","baseball glove","skateboard",
           "surfboard","tennis racket","bottle","wine glass","cup",
           "fork","knife","spoon","bowl","banana","apple","sandwich",
           "orange","broccoli","carrot","hot dog","pizza","donut",
           "cake","chair","couch","potted plant","bed","dining table",
           "toilet","tv","laptop","mouse","remote","keyboard",
           "cell phone","microwave","oven","toaster","sink",
           "refrigerator","book","clock","vase","scissors",
           "teddy bear","hair drier","toothbrush"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, size=(300, 300),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            label = f"{CLASSES[class_id]}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Nav Aid - Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()