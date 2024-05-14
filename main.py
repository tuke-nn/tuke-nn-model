import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import REAL_IMAGES_PATH, GENERATED_IMAGES_PATH, SAVE_MODEL_PATH
from images_loader import ImagesLoader


def load_and_preprocess_images(images_path):
    loader = ImagesLoader()
    images = loader.load_images(images_path)
    processed_images = preprocess_images(images)
    return processed_images


def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        processed_images.append(img)
    return np.array(processed_images)


def build_model():
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(1/.255, input_shape=(100, 100, 3)),
        layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(100, 100, 3)),
        layers.experimental.preprocessing.RandomZoom(0.1),
        layers.experimental.preprocessing.RandomContrast(0.2),
    ])
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=60, validation_data=(X_val, y_val))
    return history


def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Final Test Loss: {test_loss}, Final Test Accuracy: {test_accuracy}')
    return test_accuracy


def plot_training_history(history, final_accuracy):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.annotate(f'Final Accuracy: {final_accuracy:.2f}',
                 xy=(len(history.history['accuracy']), final_accuracy),
                 xycoords='data',
                 xytext=(-100, -20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"),
                 horizontalalignment='right',
                 verticalalignment='bottom')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Real', 'Generated'], yticklabels=['Real', 'Generated'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for Amigurumi recognition')
    plt.show()

def main():
    real_images_processed = load_and_preprocess_images(REAL_IMAGES_PATH)
    generated_images_processed = load_and_preprocess_images(GENERATED_IMAGES_PATH)

    real_labels = np.zeros(len(real_images_processed))
    generated_labels = np.ones(len(generated_images_processed))

    X = np.concatenate((real_images_processed, generated_images_processed), axis=0)
    y = np.concatenate((real_labels, generated_labels), axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1234)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=1234)

    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    model.save(SAVE_MODEL_PATH)

    final_accuracy = evaluate_model(model, X_test, y_test)
    plot_training_history(history, final_accuracy)

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    plot_confusion_matrix(y_test, y_pred_binary)


if __name__ == "__main__":
    main()
