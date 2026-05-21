
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

MODEL_PATH = "mnist_cnn_custom.h5"
TRAIN_CSV = "mnist_train.csv"   # Kaggle-style: label, p0...p783
TEST_CSV  = "mnist_test.csv"    # Kaggle-style: label, p0...p783

# Model: CNN
def build_model(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#Đọc dữ liệu
def _try_load_csv(path):
    """Đọc CSV MNIST (label ở cột 0, 784 pixel sau) mà không cần pandas.
    Trả về (X, y) với X float32 [0..1], shape (N,28,28,1).
    """
    if not os.path.exists(path):
        return None, None

    data = None
    # Thử không header
    try:
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
    except Exception:
        # Thử bỏ 1 dòng header
        try:
            data = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1)
        except Exception as e:
            print(f" Không đọc được CSV: {path}: {e}")
            return None, None

    if data.ndim != 2 or data.shape[1] < 2:
        print(f" CSV không đúng định dạng: {path}")
        return None, None

    y = data[:, 0].astype(np.int64)
    X = data[:, 1:785] if data.shape[1] >= 785 else data[:, 1:]
    if X.shape[1] != 784:
        print(f" Số cột pixel không phải 784 (thực tế {X.shape[1]}).")
        return None, None

    X = X.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0 # Chuẩn hóa dữ liệu
    return X, y

def load_data():
    # 1) Ưu tiên CSV do bạn cung cấp
    x_train, y_train = _try_load_csv(TRAIN_CSV)
    x_test,  y_test  = _try_load_csv(TEST_CSV)

    if x_train is not None and x_test is not None:
        print(" Đã nạp dữ liệu từ CSV (mnist_train.csv / mnist_test.csv).")
        return (x_train, y_train), (x_test, y_test)

    # 2) Fallback: MNIST keras
    print(" Không tìm thấy CSV hợp lệ → dùng tf.keras.datasets.mnist().")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    num_classes = int(np.max(y_train)) + 1  # mong đợi = 10
    y_train_oh = to_categorical(y_train, num_classes) # Mã hóa nhãn
    y_test_oh  = to_categorical(y_test,  num_classes)

    model = build_model(input_shape=(28,28,1), num_classes=num_classes)
    model.summary()

    # Chia dữ liệu Train/Validation/Test
    history = model.fit(
        x_train, y_train_oh,
        validation_split=0.1,
        epochs=8, batch_size=128, verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"\n Test accuracy: {acc:.4f}, loss: {loss:.4f}")

    model.save(MODEL_PATH)
    print(f" Đã lưu model: {MODEL_PATH}")

    # Demo dự đoán nhanh 5 ảnh test
    preds = model.predict(x_test[:5])
    labels = np.argmax(preds, axis=1)
    print(" Dự đoán 5 mẫu đầu tiên:", labels)

    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"P:{labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        # trong môi trường không có GUI
        pass

if __name__ == "__main__":
    main()
