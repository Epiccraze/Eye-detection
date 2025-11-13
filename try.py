import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
blink_count = 0
eyes_closed_frames = 0
EYE_CLOSED_FRAMES_THRESHOLD = 3
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30, 30))
        if len(eyes) >= 2:
            eyes_closed_frames = 0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        else:
            eyes_closed_frames += 1
            if eyes_closed_frames == EYE_CLOSED_FRAMES_THRESHOLD:
                blink_count += 1
    cv2.putText(frame, f'Blinks: {blink_count}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Face & Blink Detection', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()