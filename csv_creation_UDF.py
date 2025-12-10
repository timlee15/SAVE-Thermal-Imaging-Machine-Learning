"""
Course Number: ENGR 13300
Semester: Fall 2025

Description:
    This program evaluates the accuracy of a machine learning model that
    detects whether a child has been left behind in a vehicle based on 
    thermal imaging. The user can also test specific case numbers and
    see whether the model was correct or not.

Assignment Information:
    Assignment:     18 Individual Project UDF
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

from pathlib import Path
import pandas as pd

def classID_from_name(filename):
    # The fileset that I used contained 7 folders, with each one containing a different type of thing (ie baby, kid, teen/adult; vs empty seat, empty carseat, bag, etc.)
    # Thus, this function returns the correct classID for the images
    stem = filename.stem
    final_digit = stem[-1] # Files ended in a specific number depending on which folder they were from, which indicates their classID
    
    if final_digit in {"1", "2", "3"}:
        return 1
    elif final_digit in {"0", "4", "5", "6"}:
        return 0
    else:
        return None

def build_df(folder_path, csv_path):
    # Builds a dataframe and stores it as a csv, while also returning the dataframe for further use
    images = Path(folder_path)
    rows = []

    for img in images.glob("*"): # The .glob allows it to iterate through the folder
        if img.suffix in {".png", ".jpg", ".jpeg"}: # Ensures that it is an image file
            label = classID_from_name(img) # Calls the above UDF to get the classID
            if label is not None: # Adds the classID and the filepath to an array
                rows.append({
                    "filepath": str(img),
                    "class_id": label
                })
            else:
                print(f"Skipping {img}: Class ID does not exist.")

    df = pd.DataFrame(rows) # Converts array to dataframe for later ML
    df.to_csv(csv_path, index=False) # Creates a csv with this information stored for caching
    
    return df