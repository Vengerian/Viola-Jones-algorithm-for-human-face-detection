"""Свой способ, без использования библиотеки

С применением NMS
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_integral_image(image):
    """
    Вычисление интегрального изображения с помощью OpenCV.
    """
    return cv2.integral(image)

def haar_feature(integral_image, x, y, w, h, feature_type):
    """
    Вычисление признака Хаара для различных типов признаков.
    """
    if feature_type == "horizontal":
        mid_x = x + w // 2
        white = integral_image[y + h, mid_x] - integral_image[y, mid_x] \
                - integral_image[y + h, x] + integral_image[y, x]
        black = integral_image[y + h, x + w] - integral_image[y, x + w] \
                - integral_image[y + h, mid_x] + integral_image[y, mid_x]
        return black - white

    elif feature_type == "vertical":
        mid_y = y + h // 2
        white = integral_image[mid_y, x + w] - integral_image[mid_y, x] \
                - integral_image[y, x + w] + integral_image[y, x]
        black = integral_image[y + h, x + w] - integral_image[y + h, x] \
                - integral_image[mid_y, x + w] + integral_image[mid_y, x]
        return black - white

    return 0

def sliding_window(image, step_size, window_size):
    """
    Генератор для сканирования изображения скользящим окном.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def viola_jones_face_detection(image, window_size, step_size, threshold):
    """
    Основной метод Виолы-Джонса для обнаружения лиц с использованием дополнительных признаков.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    integral_image = compute_integral_image(gray_image)

    detected_faces = []
    for (x, y, _) in sliding_window(gray_image, step_size, window_size):
        height, width = window_size
        features = [
            haar_feature(integral_image, x, y, width, height, "horizontal"),
            haar_feature(integral_image, x, y, width, height, "vertical")
        ]

        # Среднее значение признаков
        feature = np.mean(features)

        # Условие обнаружения лица
        if feature > threshold:  # Порог для усреднённого признака
            detected_faces.append((x, y, width, height, feature))

    return detected_faces

def non_maximum_suppression(detected_faces, overlap_thresh):
    """
    Функция Non-Maximum Suppression (NMS) для устранения перекрывающихся прямоугольников.
    """
    if len(detected_faces) == 0:
        return []

    # Сортируем лица по вероятности (feature)
    detected_faces = sorted(detected_faces, key=lambda x: x[4], reverse=True)

    # Список для сохранения окончательных прямоугольников
    final_faces = []

    while len(detected_faces) > 0:
        # Берём лицо с наибольшей вероятностью
        face = detected_faces.pop(0)
        x1, y1, w1, h1, _ = face
        to_delete = []

        for i, (x2, y2, w2, h2, _) in enumerate(detected_faces):
            # Вычисление перекрытия (IoU) между прямоугольниками
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            iou = inter_area / union_area

            # Если перекрытие слишком большое, то удаляем второе лицо
            if iou > overlap_thresh:
                to_delete.append(i)

        # Удаляем перекрывающиеся прямоугольники
        detected_faces = [detected_faces[i] for i in range(len(detected_faces)) if i not in to_delete]

        # Добавляем текущее лицо в список итоговых
        final_faces.append((x1, y1, w1, h1))

    return final_faces

# Загрузка изображения
image_path = "/content/test_image9.jpg"  # Укажите путь к вашему изображению
image = cv2.imread(image_path)

# Масштабирование изображения
scale_factor = 1
resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

# Обнаружение лиц
faces = viola_jones_face_detection(resized_image, window_size=(350, 350), step_size=10, threshold=5000)

# Применяем NMS
faces_after_nms = non_maximum_suppression(faces, overlap_thresh=0.1)

# Отображение результата
for (x, y, w, h) in faces_after_nms:
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()