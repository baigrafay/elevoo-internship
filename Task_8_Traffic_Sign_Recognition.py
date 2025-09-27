import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
MobileNetV2 = tf.keras.applications.MobileNetV2
Sequence = tf.keras.utils.Sequence
layers = tf.keras.layers
models = tf.keras.models

# Set paths
data_dir = r"C:\Users\Farah\.cache\kagglehub\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1"
csv_path = os.path.join(data_dir, "Train.csv")

# Load CSV
df = pd.read_csv(csv_path)
df['path'] = df['Path'].apply(lambda x: os.path.join(data_dir, x))

# Encode labels
label_map = {label: idx for idx, label in enumerate(sorted(df['ClassId'].unique()))}
df['label'] = df['ClassId'].map(label_map)

# Split train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Custom generator
class TrafficSignSequence(Sequence):
    def __init__(self, df, batch_size, img_height, img_width, augment):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment
        self.n = len(df)
        self.indices = np.arange(self.n)
        self.aug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15 if augment else 0,
            width_shift_range=0.1 if augment else 0,
            height_shift_range=0.1 if augment else 0,
            zoom_range=0.2 if augment else 0,
            horizontal_flip=augment
        )

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_df = self.df.iloc[batch_idx]
        X = np.zeros((len(batch_df), self.img_height, self.img_width, 3), dtype=np.float32)
        y = np.zeros((len(batch_df),), dtype=np.int32)
        for i, row in enumerate(batch_df.itertuples()):
            img = tf.keras.preprocessing.image.load_img(row.path, target_size=(self.img_height, self.img_width))
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.aug.random_transform(img)
            X[i] = img / 255.0
            y[i] = row.label
        return X, tf.keras.utils.to_categorical(y, num_classes=len(label_map))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Generators
batch_size = 32
img_height, img_width = 244, 244
train_gen = TrafficSignSequence(train_df, batch_size, img_height, img_width, augment=True)
val_gen = TrafficSignSequence(val_df, batch_size, img_height, img_width, augment=False)

# Custom CNN model
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(train_gen, validation_data=val_gen, epochs=10)

# Evaluate custom CNN
y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_gen])
y_pred = cnn.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Custom CNN Classification Report:")
print(classification_report(y_true, y_pred_classes))
print("Custom CNN Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

# Transfer learning with MobileNetV2
# Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False
tl_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_map), activation='softmax')
])
tl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
tl_model.fit(train_gen, validation_data=val_gen, epochs=5)

# Evaluate MobileNetV2 model
y_pred_tl = tl_model.predict(val_gen)
y_pred_tl_classes = np.argmax(y_pred_tl, axis=1)
print("MobileNetV2 Classification Report:")
print(classification_report(y_true, y_pred_tl_classes))
print("MobileNetV2 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_tl_classes))

# Compare accuracy
acc_cnn = np.mean(y_true == y_pred_classes)
acc_tl = np.mean(y_true == y_pred_tl_classes)
print(f"Custom CNN Accuracy: {acc_cnn:.4f}")
print(f"MobileNetV2 Accuracy: {acc_tl:.4f}")
