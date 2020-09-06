import cv2

face_data = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

# img = cv2.imread('images/me.jpg')
webcam = cv2.VideoCapture(0)

while True:
    
    frame_read, frame = webcam.read()
    
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    exact_face_coordinates = face_data.detectMultiScale(gray_scaled_img)

    for (x, y, w, h) in exact_face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # print(exact_face_coordinates)

    cv2.imshow('AI Face Detector', frame)
    key = cv2.waitKey(1)
    
    if key == 81 or key == 113:
        break

webcam.release()
print("Successfully Excecuted")