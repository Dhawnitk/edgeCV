import cv2
import numpy as np
import face_recognition
from pathlib import Path
import pickle
import os
import base64
import io
from imageio import imread
import sys

# video_capture = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_DUPLEX


def load_weights(encoding_file):
    if Path(encoding_file).is_file():
        f = open(encoding_file, "rb")
        known_face_names = pickle.load(f)
        known_face_encodings = pickle.load(f)
        f.close()
        return os.stat(encoding_file).st_mtime, known_face_names, known_face_encodings
    else:
        print("No encoding file!", flush=True)


def face_tracking():

    refresh = 0
    # capture frames from the camera
    while True:
        # Grab a single frame of video
        frame_read = input()
        # line = "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"
        line = ""
        if frame_read == "ok":
            with open('/home/root1/Desktop/temp.txt') as fp:
                for line1 in fp:
                    line = line1

        img = imread(io.BytesIO(base64.b64decode(line)))
        # ret, frame = video_capture.read()
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        encoding_file = "/home/root1/Desktop/EdgeCV/edgeCV/known_face_encodings.p"
        cached_stamp, known_face_names, known_face_encodings = load_weights(encoding_file)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, 0.5)
            name = "Unknown"
            if True in matches:
                if matches.count(True) > 1:
                    first_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
                else:
                    first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()

        refresh += 1
        if refresh == 1:
            # video_capture.release()
            break

        # print (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        frame_serialize = base64.b64encode(frame).decode("utf-8")
        print (frame_serialize, flush=True)

face_tracking()