__author__ = 'jiang'
import glob
import  numpy as np
import csv
from skimage import img_as_float
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def read_data(typeData, imageSize,trainID , path):
    x = np.zeros((len(trainID), imageSize))
    trainFiles = glob.glob( path + "/"+typeData + "Resized/*" )
    for i, nameFile in enumerate(trainFiles):
        # Read image
        nameFile = "/".join( nameFile.split("/")[:-1] ) +"/" + trainID[i]+ "."+nameFile.split(".")[-1]

        image = imread(nameFile)
        if( i==2289):
            print i
        if( len(image.shape)==3):
            # Convert to float
            temp = img_as_float(image)
            # Conver to gray image
            gray_image = rgb2gray(temp)
        else:
            gray_image = np.divide(image[:, :], 255.0)
        # Transform image matrix to vector
        x[i , : ]=  np.reshape(gray_image,(1 ,imageSize))

    return x

def read_csv(file):
    id = []
    classes = []
    with open(file,'r') as csvFile:
       read_data = csv.DictReader(csvFile,delimiter=',')
       for row in read_data:
           id.append( str(row['ID']))
           classes.append(ord(row['Class']))

    return (np.array(id), np.array(classes))

def write_csv(file, x,y):
    with open(file, 'w') as csvFile:
        fieldnames = ["ID","Class"]
        writer = csv.DictWriter(csvFile,  fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(x)):
            writer.writerow({"ID": x[i], "Class": y[i]})


def decisiont_tree_classify(x, y, xTest):
    clf = RandomForestClassifier(n_estimators=50, max_features=20)
    clf = clf.fit(x,y)
    yPredict =  clf.predict(xTest)
    return map((lambda x:chr(x) ), yPredict)

def svm_classify(x, y, xTest):
     clf = svm.SVC()
     clf.fit(x,y)
     yPredict =  clf.predict(xTest)
     return map((lambda x:chr(x) ), yPredict)



imageSize = 400 # using image size 20*20
path = '.'

# Read in train labels
trainID, yTrain = read_csv("trainLabels.csv")
xTrain = read_data("train", imageSize, trainID, path)

# Read in test labels
testID, yTest = read_csv("testLabels.csv")
xTest = read_data("test", imageSize, testID,path)

yPredict = decisiont_tree_classify(xTrain, yTrain, xTest)
#yPredict = svm_classify(xTrain, yTrain,xTest)
print yPredict
write_csv("testLabels.csv", testID, yPredict)