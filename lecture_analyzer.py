# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

import matplotlib.pyplot as plt

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_text(img, text, coord_x, coord_y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45

    rectangle_bgr = (0, 0, 0)
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((coord_x - 5, coord_y + 5), (coord_x + text_width + 5, coord_y - text_height - 5))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)

    cv2.putText(img, text, (coord_x, coord_y), font, fontScale=font_scale, color=(225, 225, 225), thickness=1)


def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]

    return (x, y)


face_model = get_face_detector()
landmark_model = get_landmark_model()
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

start_time = time.monotonic()
max_len_faces = 0
inattentive_accum = 0.0

fig = plt.figure(figsize=(13, 4.8))
plt.ylim(-0.1, 1.1)
plt.xlabel("Lecture time (s)")
plt.ylabel(r'Attention factor (1 - $\frac{inattentiveTime}{allStudents * lectureTime}$)')

plt.axhline(y=1, color='#54cb0b', linestyle='--', linewidth=1, label='max. attention')
plt.axhline(y=0, color='#f52922', linestyle='--', linewidth=1, label='max. inattention')

# plt.ylabel("Inattention factor (inattentive_time / (all_students * lecture_time))" r'Inattention factor $\frac{inattentive_time - {all_students * lecture_time}$')

x = []
y = []

every_second = 0
legend = None
text_plot = None

while True:
    ret, img = cap.read()
    if ret == True:
        faces = find_faces(img, face_model)

        if len(faces) > max_len_faces:
            max_len_faces = len(faces)

        for face in faces:
            confidence = face["confidence"]
            coordinates = face["coordinates"]
            attention = face["attention"]
            xx, yy, xxx, yyy = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
            marks = detect_marks(img, landmark_model, coordinates)

            image_points = np.array([
                marks[30],  # Nose tip
                marks[8],  # Chin
                marks[36],  # Left eye left corner
                marks[45],  # Right eye right corne
                marks[48],  # Left Mouth corner
                marks[54]  # Right mouth corner
            ], dtype="double")
            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            try:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90

            try:
                m = (x2[1] - x1[1]) / (x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1 / m)))
            except:
                ang2 = 90

            if (ang1 >= 45 or ang1 <= -45) or (ang2 >= 45 or ang2 <= -45):
                face.update({"attention": 0})
                if int(time.monotonic() - start_time) != every_second:
                    inattentive_accum += 1
                cv2.rectangle(img, (xx, yy), (xxx, yyy), (34, 41, 245), 1)
            else:
                face.update({"attention": 1})
                cv2.rectangle(img, (xx, yy), (xxx, yyy), (11, 203, 84), 1)

            draw_text(img, "Pitch: {0} Yaw: {1}".format(str(ang2), str(ang1)), xx, yyy + 20)
            draw_text(img, "Face: {:.2f}%".format(confidence * 100), xx, yy - 10)
            draw_text(img, "Attentive: {0}".format(str(bool(face["attention"]))), xxx + 10, yy + 10)

        draw_text(img, "Lecture time (seconds): {0}".format(str(int(time.monotonic() - start_time))), 30, 30)
        draw_text(img, "Max. faces on the screen: {0}".format(str(max_len_faces)), 30, 50)
        draw_text(img, "Listeners on the screen: {0}".format(str(len(faces))), 30, 70)

        attentive_listeners = len([i for i in faces if i["attention"] == 1])
        inattentive_listeners = len([i for i in faces if i["attention"] == 0])

        draw_text(img, "Attentive listeners: {0}".format(str(int(attentive_listeners))), 30, 90)
        draw_text(img, "Inattentive listeners: {0}".format(str(int(inattentive_listeners))), 30, 110)
        draw_text(img, "Inattentive time (seconds): {0}".format(str(int(inattentive_accum))), 30, 130)

        if int(time.monotonic() - start_time) != every_second:
            every_second = int(time.monotonic() - start_time)

            meter = inattentive_accum
            denominator = len(faces) * int(time.monotonic() - start_time)
            if denominator != 0:
                x.append(int(time.monotonic() - start_time))
                y.append(1 - (meter/denominator))
                if text_plot != None:
                    text_plot.set_visible(False)
                text_plot = plt.text(x[-1], y[-1], str(round(1 - (meter / denominator), 2)), fontsize=8, color='#0b68ff')
                plt.plot(x, y, linewidth=1, color='#0b68ff', label='Attention factor')
                if legend == None:
                    legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
                fig.canvas.draw()
                graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                graph = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)
                cv2.imshow("Attention factor graph", graph)

        cv2.imshow('Lecture analyzer', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cv2.destroyAllWindows()
cap.release()
