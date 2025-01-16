!pip install opencv-python opencv-python-headless

import cv2
import matplotlib.pyplot as plt

# Загрузка классификатора Виолы-Джонса
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загрузка изображения
image_path = 'test_image9.jpg'  # Укажите путь к вашему изображению
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Конвертация в градации серого

# Детекция лиц
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))

# Отображение результата(рисование прямоугольника вокруг лица)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Покажем изображение
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Увеличение контраста
gray_image = cv2.equalizeHist(gray_image)# выравнивание гистограммы яркости для улучшения контраста изображения.

# Уменьшение шума
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)#применяет фильтр Гаусса для сглаживания изображения и удаления мелкого шума.

# Детекция лиц
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Отображение результата
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Покажем изображение
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()