import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

## Function to check directory existence and permission
def check_directory(path):
    if os.path.exists(path):
        print(f"Directory exist: {path}")
        if os.access(path, os.R_OK):
            print(f"Directory is readable: {path}")
            
        else:
            print(f"Directory is not readable: {path}")
            
## Print the currect working directory 
print(f"Current working directory: {os.getcwd()}")

# prepare data
input_dir = r'C:\Users\Fame\Desktop\Projects_2024\Computer_Vision_Beginner_Projects\ImageClassification_Python_Scikit-Learn\clf-data'
categories = ['empty', 'non_empty']

# Check if the input_dir exists and is accessible
check_directory(input_dir)

data = []
labels = []
for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    ## check if each category directory exists and is accessible
    check_directory(category_path)
    
    if not os.path.exists(category_path):
        print(f"Directory does not exist: {category_path}")
        continue
    for file in os.listdir(category_path):
        img_path=os.path.join(category_path, file)
        print(f"Processing File: {img_path}") ## Print the file being processed
        try:
            img = imread(img_path)
            img = resize(img, (15, 15))
            data.append(img.flatten())
            labels.append(category_idx)

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
        
       

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters)

grid_search.fit(x_train, y_train)

# test performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('.\\model.p', 'wb'))