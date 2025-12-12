"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    This program evaluates the accuracy of a machine learning model that
    detects whether a child has been left behind in a vehicle based on 
    thermal imaging. The user can also test specific case numbers and
    see whether the model was correct or not.

Assignment Information:
    Assignment:     18 Individual Project
    Team ID:        LC3 - 24
    Author:         Timothy Lee, lee5695@purdue.edu
    Date:           11/24/2025

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

# Citation (for the dataset that I used):
# Authors: Dias Da Cruz, S., Wasenmüller, O., Beise, H.-P., Stifter, T., & Stricker, D.
# Year: 2020
# Title: SVIRO: Synthetic Vehicle Interior Rear Seat Occupancy Dataset and Benchmark.
# Book Title: In IEEE Winter Conference on Applications of Computer Vision (WACV).

from csv_creation_UDF import build_df
from csv_creation_UDF import classID_from_name
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import hog
import cv2

def find_folder(path):
    # This function determines whether the path to the folder exists, and is_dir ensures that it is a folder (not just a file)
    folder_path = Path(path.strip())
    if folder_path.exists() and folder_path.is_dir():
            return folder_path
    else:
        return None

def load_csv(train_csv, train_path, test_csv, test_path):
    # This function checks whether the requested csv file already exists
    if train_csv.exists(): # If the file exists, it loads it as a Pandas dataframe
        train_df = pd.read_csv(train_csv)
        print(f"\nThe file {train_csv} was found and has been successfully loaded.")
    else: # If not, it uses build_df, which is called from another file, to make the dataframe
        print(f"\nThe file {train_csv} does not already exist.")
        print(f"Extracting ClassIDs to create the file {train_csv}...")
        train_df = build_df(train_path, train_csv)
        print(f"Saved the file as {train_csv} with {len(train_df)} entries.\n")

    # Same as above
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        print(f"The file {test_csv} was found and has been successfully loaded.")
    else:
        print(f"The file {test_csv} does not already exist.")
        print(f"Extracting ClassIDs to create the file {test_csv}...")
        test_df = build_df(test_path, test_csv)
        print(f"Saved the file as {test_csv} with {len(test_df)} entries.")

    return train_df, test_df

def HOG_feature_extraction(image_paths, size=(64, 128)):
    # This function creates ONE Numpy array containing every (row) feature vector for all images in the folder
    features = []
    for path in image_paths: # Uses the cv2 library to read the images and resize them
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        resized_img = cv2.resize(img, size)
        # Creates Histogram of Oriented Gradients using sci-kit learn and a bunch of fancy math, resulting in row vectors of length 7*15*36 = 3780 columns
        hog_feature_vector = hog(
                                resized_img,
                                orientations=9,
                                pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2),
                                block_norm='L2-Hys',
                                transform_sqrt=True
        )
        features.append(hog_feature_vector) # Appends row vector to the features list; array has the same number of rows as images in the folder
    return np.array(features)

def train_SVM(X_train, y_train, k_type):
    # This function creates the support vector machine model based on the data
    svm_model = SVC(
                    kernel=k_type,
                    C=0.25,
                    class_weight="balanced",
                    probability=True
    ) # Creates an empty model with the desired parameters
    svm_model.fit(X_train, y_train) # Fits the model to the data (finds a hyperplane)
    return svm_model

def eval_results(model, X_test, y_test):
    # This function evaluates the effectiveness of the model compared to the test data
    predictions = model.predict(X_test) # Using the feature vectors for X_test, it plots these and based on the side of the hyperplane, they are classified

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions)) # Function from scikit learn that compares the list of predictions to the correct ones y_test

    return None

def single_prediction(folder, model):
    # This function returns whether the prediction was right for a specific image
    folder_path = Path(folder)
    image_files = []
    for file in folder_path.iterdir(): # Creates a list of all the image names
        image_files.append(file)
    print(f"\nThere are {len(image_files)} images in this folder.")

    while True: # While loop used here to error check and ensure that input is valid, repeating until it is
        number = input(f"Of the {len(image_files)} files, which one would you like to test? ")
        if not number.isdigit():
            print('Error: That is not a valid file number. Please try again.')
            continue
        number = int(number)
        if number < 1 or number > len(image_files):
            print('Error: That is not a valid file number. Please try again.')
            continue
        else:
            index = number - 1
            img_path = image_files[index]
            correct_ID = classID_from_name(img_path) # Calls a function from the other file to check just the single classID
            break
    
    features = HOG_feature_extraction([img_path]) # Calls the HOG feature extraction for just this file, in brackets because it expects an array
    prediction = model.predict(features)[0] # Using that feature vector, it makes a prediction

    if prediction == correct_ID:
        print("\nThe model was CORRECT.")
    else:
        print("\nThe model was INCORRECT.")

    return None

def main():
    # Get user input for the name of the csv file and the folder
    train_csv = input("Enter the path to the TRAINING csv file: ")
    train_csv = Path(train_csv)
    train_path = None
    while train_path == None: # While loop for error checking until the folder inputted exists
        train_path = input("Enter the path to the TRAINING image folder: ")
        train_path = find_folder(train_path)
        if train_path is None:
            print("Error: The folder does not exist. Please try again.")

    # Same as above
    test_csv = input("\nEnter the path to the TESTING csv file: ")
    test_csv = Path(test_csv)
    test_path = None
    while test_path == None:
        test_path = input("Enter the path to the TESTING image folder: ")
        test_path = find_folder(test_path)
        if test_path is None:
            print("Error: The folder does not exist. Please try again.")

    # Calls the load_csv function to either bring in the cached csvs, or create them
    train_df, test_df = load_csv(train_csv, train_path, test_csv, test_path)

    # They are returned as dataframes, so they are separated based on their columns
    train_paths = train_df["filepath"]
    train_labels = train_df["class_id"]

    test_paths = test_df["filepath"]
    test_labels = test_df["class_id"]

    # Extracts the features using the paths of all the images in the folder, converts the classIDs into Numpy array
    print('\nExtracting TRAINING features via Histogram of Oriented Gradients...')
    X_train = HOG_feature_extraction(train_paths)
    y_train = np.array(train_labels)
    print('TRAINING feature extraction complete.')

    # Same as above
    print('\nExtracting TESTING features via Histogram of Oriented Gradients...')
    X_test = HOG_feature_extraction(test_paths)
    y_test = np.array(test_labels)
    print('TESTING feature extraction complete.')

    # Asks the user which kernel type they would like to use to create the SVM model
    valid_kernels = ["linear", "rbf", "poly", "sigmoid"]
    while True: # Error checking to ensure that the inputted kernel type is one of the options
        k_type = input("\nChoose which SVM kernel type to use (linear, rbf, poly, sigmoid): ").lower().strip()
        if k_type in valid_kernels:
            break
        else:
            print(f"Error: Invalid kernel. Please try again.")

    # Trains the model
    print('\nTraining Support Vector Machine model and finding the best hyperplane...')
    model = train_SVM(X_train, y_train, k_type)
    print('Model completed.')

    # Calls UDF to display the results of the model
    eval_results(model, X_test, y_test)

    # Gets user input for the folder to pull the test cases from (this is for future integration if you want to extrapolate the data to a different vehicle)
    # At this time that accuracy is kind of bad because the training data exclusively uses ~5000 images from the BMW i3
    folder_path = None
    while folder_path == None:
        folder_path = input("Enter the name of the folder with the possible test cases: ")
        folder_path = find_folder(folder_path)
        if folder_path is None:
            print("Error: The folder does not exist. Please try again.")

    # Calls the function to make a prediction
    single_prediction(folder_path, model)
    
    # While loop here with error checking to run the single prediction again because it's really annoying having to run the whole code all
    # over again just to try and make a second prediction
    while True:
        run_again = input('\nWould you like to run the single prediction again? (y/n) ')
        if run_again == 'y':
            single_prediction(folder_path, model)
        elif run_again == 'n':
            break
        else:
            print('Error: Invalid input, please try again.')

if __name__ == "__main__":
    main()