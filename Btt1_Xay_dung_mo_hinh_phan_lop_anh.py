
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

def read_img_data(path, label, size):
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

def main():
    X,y = read_img_data('A:/Study/HocMay/Đồ án/PetImages/Cat','cat', (32,32))
    X_dog, y_dog = read_img_data('A:/Study/HocMay/Đồ án/PetImages/Dog','dog', (32,32))
    X = np.extend(X_dog)
    y = np.extend(y_dog)
    X = np.array(X)
    y = LabelBinarizer().fit_transform(y)
    print("x shape", X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state =1)

    #LogisticRegression Model
    model = LogisticRegression()
    # Huấn luyện mô hình với tập dữ liệu X, y
    model.fit(X, y)
    print('Trọng số tối ưu', model.coef_)

    #K-NN Model


if __name__ == '__main__':
    main()