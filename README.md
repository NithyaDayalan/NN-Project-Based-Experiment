# Project Based Experiments
### NAME : NITHYA D
### REG.NO : 212223240110
## Objective :
 Build a Multilayer Perceptron (MLP) to classify handwritten digits in python
## Steps to follow :
## Dataset Acquisition :
Download the MNIST dataset. You can use libraries like TensorFlow or PyTorch to easily access the dataset.
## Data Preprocessing :
Normalize pixel values to the range [0, 1].
Flatten the 28x28 images into 1D arrays (784 elements).
## Data Splitting :

Split the dataset into training, validation, and test sets.
Model Architecture:
## Design an MLP architecture. 
You can start with a simple architecture with one input layer, one or more hidden layers, and an output layer.
Experiment with different activation functions, such as ReLU for hidden layers and softmax for the output layer.
## Compile the Model :
Choose an appropriate loss function (e.g., categorical crossentropy for multiclass classification).Select an optimizer (e.g., Adam).
Choose evaluation metrics (e.g., accuracy).
## Training :
Train the MLP using the training set.Use the validation set to monitor the model's performance and prevent overfitting.Experiment with different hyperparameters, such as the number of hidden layers, the number of neurons in each layer, learning rate, and batch size.
## Evaluation :

Evaluate the model on the test set to get a final measure of its performance.Analyze metrics like accuracy, precision, recall, and confusion matrix.
## Fine-tuning :
If the model is not performing well, experiment with different architectures, regularization techniques, or optimization algorithms to improve performance.
## Visualization :
Visualize the training/validation loss and accuracy over epochs to understand the training process. Visualize some misclassified examples to gain insights into potential improvements.

## Program :
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

EPOCHS = 15
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

print("Loading MNIST dataset...")
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Calculate the split index and ensure it is an integer
split_index = int(VALIDATION_SPLIT * x_train_full.shape[0])

# Use the integer index for slicing
x_train, x_valid = x_train_full[split_index:], x_train_full[:split_index]
y_train, y_valid = y_train_full[split_index:], y_train_full[:split_index]

x_train = x_train / 255.0 # Normalize pixel values
x_valid = x_valid / 255.0
x_test = x_test / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Validation samples: {x_valid.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

model = Sequential([
    Flatten(input_shape=(28, 28)), # Input layer (784 neurons)
    
    Dense(512, activation='relu'),
    Dropout(0.2), # Add regularization
    
    Dense(256, activation='relu'),
    Dropout(0.2),
    
    Dense(10, activation='softmax') # Output layer (10 classes)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

print("\nStarting model training...")
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_valid, y_valid),
    verbose=1
)

print("\nEvaluating model on the test set...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Final Test Loss: {test_loss:.4f}")
print(f"✅ Final Test Accuracy: {test_acc:.4f}")

def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, x_test, y_test):
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix for MNIST Test Set')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.show()

def plot_misclassified_images(model, x_test, y_test, num_images=5):
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    misclassified_indices = np.where(y_pred != y_test)[0]
    
    plt.figure(figsize=(15, 3 * num_images))
    plt.suptitle("Misclassified Examples", fontsize=16)

    for i in range(min(num_images, len(misclassified_indices))):
        index = misclassified_indices[i]
        plt.subplot(1, num_images, i + 1)
        plt.imshow(x_test[index], cmap='gray')
        plt.title(f"Pred: {y_pred[index]} \n True: {y_test[index]}", color='red')
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

plot_history(history)
plot_confusion_matrix(model, x_test, y_test)
plot_misclassified_images(model, x_test, y_test, num_images=5)

print("\nModel training and analysis complete.")
```

## Output :
<img width="1724" height="422" alt="image" src="https://github.com/user-attachments/assets/0d66f9ec-5434-4194-9b5c-8f78a1a506b5" />
<img width="887" height="596" alt="image" src="https://github.com/user-attachments/assets/cd82ba25-9ea1-4c55-bdfa-5d1846eca026" />
<img width="1202" height="407" alt="image" src="https://github.com/user-attachments/assets/09cc3c25-25ee-497b-8835-fa114b65453b" />
<img width="800" height="625" alt="image" src="https://github.com/user-attachments/assets/c8650fb8-3310-4554-9ccb-50490f15c120" />

## RESULT :
The Python program to build Multilayer Perceptron (MLP) to classify handwritten digits is executed and implemented successfully.
