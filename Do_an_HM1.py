
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import io
from skimage.transform import resize
import os

# Tien xu ly du lieu
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# Huan luyen mo hinh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
# Danh gia mo hinh
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

#Ham kiem tra anh co loi hay khong
def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            img_new = io.imread(os.path.join(img_file))
        return False
    except Exception as e:
        print(e)
        return True

#Ham doc file anh
def read_img_datasets(path, label, size):
    X = []
    y = []
    files = os.listdir(path)
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray=True)
            img = resize(img, size)
            img_vector = img.flatten()
            X.append(img_vector)
            y.append(label)
    X = np.array(X)
    return X,y

def encode_label(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y).reshape(y.shape[0], )

#Ham chuyen anh man hinh thanh vector 1024
def convert_D_2_vector(path, label, size):
    labels = []
    img_data = []
    images = os.listdir(path)
    for img_file in images:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img_grey = io.imread(os.path.join(path,img_file),as_gray=True)
            img_vector = resize(img_grey, size).flatten()
            img_data.append(img_vector)
            labels.append(label)

#Huan luyen mo hinh
#LogisticRegressionCV
def logistic_regression_cv(X_train, y_train):
    logistic_classifier = LogisticRegressionCV(cv=5, solver="sag", max_iter=2000)
    logistic_classifier.fit(X_train, y_train)
    return logistic_classifier
#K-NN
def kNN_grid_search_cv(X_train, y_train):
    from math import sqrt
    m = y_train.shape[0]
    k_max = int(sqrt(m) / 2)
    k_values = np.arange(start=1, stop=k_max + 1, dtype=int)
    params = {'n_neighbors': k_values}
    kNN = KNeighborsClassifier()
    kNN_grid = GridSearchCV(kNN, params, cv=3)
    kNN_grid.fit(X_train, y_train)
    return kNN_grid

def evaluate_model(y_test, y_pred):
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("Balanced accuracy score: ", balanced_accuracy_score(y_test, y_pred))
    print("Hamming loss: ", hamming_loss(y_test, y_pred))


def Confusion_Matrix(y_test, y_pred, model):
    ax1 = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", cmap="crest")
    ax1.xaxis.tick_top()
    plt.savefig("CM.png")
    plt.show()
    plt.close()


def test_score(y_test, y_pred, class_type="binary"):
    if class_type == "binary":
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        result = [accuracy, precision, recall, f1]
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        result = [accuracy, precision, recall, f1]

    return result


def test_table(modals: dict):
    result = {
        "Scores": ["accuracy", "precision", "recall", "f1"]
    }
    for modal in modals:
        modal_score = test_score(modals[modal][0], modals[modal][1])

    return pd.DataFrame(result)

def main():
    #Nhap tap du lieu
    X,y = read_img_datasets('A:/Study/HocMay/Đồ án/PetImages/Cat','cat', (32,32))
    X_dog, y_dog = read_img_datasets('A:/Study/HocMay/Đồ án/PetImages/Dog','dog', (32,32))
    X = np.extend(X_dog)
    y = np.extend(y_dog)
    X = np.array(X)
    y = LabelBinarizer().fit_transform(y)
    print("x shape", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state =1)

    # Huấn luyện mô hình:

    kNN_classifier = kNN_grid_search_cv(X_train, y_train)
    logistic_classifier = logistic_regression_cv(X_train, y_train)

    # Dự đoán kết quả:
    y_pred_kNN = kNN_classifier.predict(X_test)
    y_pred_logistic = logistic_classifier.predict(X_test)

    # Đánh giá mô hình:
    evaluate_model(y_test, y_pred_kNN)
    evaluate_model(y_test, y_pred_logistic)

if __name__ == '__main__':
    main()