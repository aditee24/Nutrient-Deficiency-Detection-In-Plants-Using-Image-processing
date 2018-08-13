import cv2
import numpy as np
import argparse
import sys
from numpy.core.multiarray import ndarray
import skimage
from openpyxl import Workbook
from skimage import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
import openpyxl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

class Segment:
    def __init__(self, segments=30):
        # define number of segments, with default
        self.segments = segments

    def kmeans(self, a):
        image = cv2.GaussianBlur(a, (5, 5), 0)
        vectorized = image.reshape(-1, 3)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(vectorized, self.segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label):
        component = np.zeros(image.shape, np.uint8)
        image[label_image == label] = component[label_image == label]
        return image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-n", "--segments", required=False, type=int,
                    help="# of clusters")
    args = vars(ap.parse_args())
    image1 = cv2.imread(args["image"])
    cv2.imshow('Input image of leaf', image1)
    image = cv2.cvtColor(image1,cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image)
   # cv2.imshow('LAB image',image)

    if len(sys.argv) == 3:
        seg = Segment()
        label, result = seg.kmeans(image)
    else:
        seg = Segment(args["segments"])
        label, result = seg.kmeans(image)

    # display segmented and extracted image
    #cv2.imshow("segmented", result)
    result1 = seg.extractComponent(image, label, 2)
    cv2.imshow("extracted", result1)

    var = np.var(result)  # type: ndarray
    m, s = cv2.meanStdDev(result)
    print('Mean is:  ', m)
    print('variance is: ', var)
    print('std dev is:  ', s)

    M1 = float(m[1])
    M2 = float(m[2])
    S1 = float(s[1])
    S2 = float(s[2])

    # glcm matrix

    lab_rgb = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    cv2.imshow('rgb', lab_rgb)
    rgb_gray = img_as_ubyte(skimage.color.rgb2gray(lab_rgb))

    # g = greycomatrix(image, [0, 1], [0, np.pi/2], levels=256)
    g = skimage.feature.greycomatrix(rgb_gray, [1], [0], levels=256, symmetric=False,
                                     normed=True)

    contrast = greycoprops(g, 'contrast')
    print('contrast is: ', contrast)

    energy = greycoprops(g, 'energy')
    print('energy is: ', energy)

    homogeneity = greycoprops(g, 'homogeneity')
    print('homogeneity is: ', homogeneity)

    correlation = greycoprops(g, 'correlation')
    print('correlation is: ', correlation)

    dissimilarity = greycoprops(g, 'dissimilarity')
    print('dissimilarity is: ', dissimilarity)

    ASM = greycoprops(g, 'ASM')
    print('ASM is: ', ASM)

    con = float(contrast)
    ene = float(energy)
    hom = float(homogeneity)
    cor = float(correlation)
    dis = float(dissimilarity)
    A = float(ASM)
    features = [M1, M2, S1, S2, var, contrast, energy, homogeneity, correlation, dissimilarity, ASM]
    feature = np.reshape(features, (1, -1))

    #  Machine learning code
    leafdata = pd.read_csv('leafdemo.csv')

    title = list(leafdata.keys())
    title_class = title[:-1]
    leaf = pd.DataFrame(leafdata, columns=title)
    X = leaf.drop('class', axis=1)
    y = leaf.drop(columns=title_class, axis=1)
    labled_class = pd.factorize(leafdata['class'])
    leafdata['class'] = labled_class[0]
    definitions = labled_class[1]
    #labled_class.fit(y)
    #scaler = StandarScaler()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.65,random_state=None)

    clf = KNeighborsClassifier(n_neighbors=4)
    clf.fit(X, y)

    predictions = clf.predict(X_test)
    y_pred = clf.predict(feature)
    print('predicted class: ', y_pred)
    acc = accuracy_score(y_test,predictions)
    print('Accuracy: ', acc)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
