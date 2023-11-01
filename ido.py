import cv2
import numpy as np

# Initialize the video capture (you can also use a video file)
cap = cv2.VideoCapture('in.avi')

# Create a background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Initialize variables for counting and tracking
persons = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to reduce noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (5, 5))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2

            # Check if this person is a new or existing person based on centroid
            new_person = True
            for person in persons:
                px, py = person['centroid']
                if np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < 50:  # Adjust the distance threshold
                    new_person = False
                    person['centroid'] = (cx, cy)
                    break

            if new_person:
                persons.append({'centroid': (cx, cy)})

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the count on the frame
    count = len(persons)
    cv2.putText(frame, "People Count: " + str(count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("People Counter", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
