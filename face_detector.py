import cv2

face_data = cv2.CascadeClassifier('opencv/haarcascade_frontalface_default.xml')

# img = cv2.imread('images/me.jpg')
webcam = cv2.VideoCapture(0) # Capture the video from the default webcam which is 0

while True: # Infinate Loop
    
    frame_read, frame = webcam.read() # Read the video and return values to two variables
    
    gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Take the video and convert it in to gray color, It will easy to read

    exact_face_coordinates = face_data.detectMultiScale(gray_scaled_img) # this will return exact face coordinates

    for (x, y, w, h) in exact_face_coordinates: # This will loop through the faces and draw the rectangle according to their coordinates
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) # This represent ( scr (x Coordinates, y Coordinates), (x + width, y + height) (B, G, R), stroke)

    # print(exact_face_coordinates)

    cv2.imshow('AI Face Detector', frame)
    key = cv2.waitKey(1)
    
    # set up ASSIC values for quit character q or Q
    # ASSIC 
        # q = 81
        # Q = 113 
    if key == 81 or key == 113: # This will terminate the program when q is pressed
        break

webcam.release()
print("Successfully Excecuted")