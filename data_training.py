import os  
import numpy as np 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

# Initialize necessary variables
is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Debugging: Check which files are in the current directory and subdirectories
print("Files in directory and subdirectories:")
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        print(f"Found file: {os.path.join(dirpath, filename)}")

# Loop through the files in the current directory and subdirectories
for dirpath, dirnames, filenames in os.walk('.'):
    for filename in filenames:
        if filename.endswith(".npy") and not(filename.startswith("labels")):  
            print(f"Processing file: {filename}")  # Debugging: Print the current file being processed
            file_path = os.path.join(dirpath, filename)
            
            # Load the .npy file and process data
            if not(is_init):
                is_init = True 
                X = np.load(file_path)
                print(f"Loaded {filename}, shape of X: {X.shape}")  # Debugging: Print shape of X
                size = X.shape[0]
                y = np.array([filename.split('.')[0]]*size).reshape(-1,1)
            else:
                new_X = np.load(file_path)
                print(f"Loaded {filename}, shape of new_X: {new_X.shape}")  # Debugging: Print shape of new_X
                new_size = new_X.shape[0]
                X = np.concatenate((X, new_X))
                y = np.concatenate((y, np.array([filename.split('.')[0]]*new_size).reshape(-1,1)))

            # Keep track of the labels and their corresponding integer values
            label.append(filename.split('.')[0])
            dictionary[filename.split('.')[0]] = c  
            c = c + 1

# Debugging: Check if y is defined correctly after data loading
if 'y' not in locals():
    print("Error: y is not defined. Check your data loading process.")
else:
    print(f"Data loaded successfully. y shape: {y.shape}")

# Convert the labels to integer values based on the dictionary
if 'y' in locals():
    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]
    y = np.array(y, dtype="int32")

    # Convert the labels to one-hot encoding
    y = to_categorical(y)

    # Create a shuffled version of the dataset
    X_new = X.copy()
    y_new = y.copy()
    counter = 0 

    # Shuffle the data
    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)

    # Reorder the data based on the shuffled indices
    for i in cnt: 
        X_new[counter] = X[i]
        y_new[counter] = y[i]
        counter = counter + 1

    # Define the model architecture
    ip = Input(shape=(X.shape[1],))
    m = Dense(512, activation="relu")(ip)
    m = Dense(256, activation="relu")(m)
    op = Dense(y.shape[1], activation="softmax")(m) 

    model = Model(inputs=ip, outputs=op)

    # Compile the model
    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

    # Train the model
    model.fit(X_new, y_new, epochs=50)

    # Save the trained model and the labels
    model.save("model.h5")
    np.save("labels.npy", np.array(label))
