import os
import cv2
from sklearn import svm
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import xgboost
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


#Папки с картинками
folder_0 = "./drunk_images/"
folder_1 = "./sober_images/"
folder_2 = "./noise_images/"
dataset = []
labels = []

#Считывание датасета и папки с картинками
def create_dataset(folder, class_name):
    image_directories = folder
    images = os.listdir(image_directories)
    for image_name in images:
        image = cv2.imread(image_directories + image_name)
        #Перевод 3 канального изображения в 1 канальное
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.array(image).flatten()
        #Заполнение датасета
        dataset.append(image)
        labels.append(class_name)

create_dataset(folder_0, 0)
create_dataset(folder_1, 1)
create_dataset(folder_2, 2)

dataset = np.array(dataset)
label = np.array(labels)
logger.info(f"Photo processing is done. Shape {dataset.shape}")

# dataset = PCA(n_components=2).fit_transform(dataset)
dataset = TSNE(random_state=42).fit_transform(dataset)

#Разбиение модели на тестовые тренировочные данные
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=42)

#Инициализация модели SVM
model = svm.SVC()
#Обучение модели
model.fit(X_train, y_train)
decision_function = model.decision_function(X_train)

prediction = model.predict(X_test)
logger.info(f"Accuaracy: {accuracy_score(y_test, prediction)}")
# plot_confusion_matrix(model, X_test, y_test)  
# plt.show()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction, alpha=0.7, marker="*")
markers = ["1", "+", "x"]
for count, val in enumerate(y_test):
    plt.scatter(X_test[count, 0], X_test[count, 1], c=val, alpha=0.7, marker=markers[val])
plt.show()

#Сохранение модели в байткод черел Pickle
model_file = "svm_model_1.sav"
pickle.dump(model, open(model_file, "wb"))
logger.info("Model is saved")

# #XGboost
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
logger.info("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot_confusion_matrix(model, X_test, y_test)  
# plt.show() 

model_file = "xgboost_model_1.sav"
pickle.dump(model, open(model_file, "wb"))
logger.info("Model is saved")

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=prediction, alpha=0.7, marker="*")
plt.show()