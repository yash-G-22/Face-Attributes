import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

os.makedirs('models', exist_ok=True)

# ---------- Load & preprocess ----------
data = pd.read_csv('age_gender.csv')
data = data.drop('img_name', axis=1)
data['age'] = pd.qcut(data['age'], q=4, labels=[0, 1, 2, 3])

target_columns = ['gender', 'ethnicity', 'age']
y = data[target_columns]

pixels = data['pixels']
X_pixels = pixels.str.split().apply(lambda s: np.fromiter(map(int, s), dtype=np.uint8))
X = np.vstack(X_pixels.values)
X = X.reshape(-1, 48, 48, 1).astype(np.float32)  # keep 0-255 for augmentation/rescaling layer

y_gender = y['gender'].astype(np.int32).values
y_ethnicity = y['ethnicity'].astype(np.int32).values
y_age = y['age'].astype(np.int32).values

# Use same train/test split for reproducibility
idx_train, idx_test = train_test_split(np.arange(len(X)), train_size=0.7, random_state=42, shuffle=True)
X_train, X_test = X[idx_train], X[idx_test]
y_gender_train, y_gender_test = y_gender[idx_train], y_gender[idx_test]
y_ethnicity_train, y_ethnicity_test = y_ethnicity[idx_train], y_ethnicity[idx_test]
y_age_train, y_age_test = y_age[idx_train], y_age[idx_test]

print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# ---------- Helper: improved model ----------
def build_improved_model(num_classes, problem='multiclass'):
    """
    problem: 'binary' for gender, 'multiclass' for ethnicity/age
    """
    inputs = tf.keras.Input(shape=(48, 48, 1))
    # Rescale in-model and augment
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomTranslation(0.06, 0.06)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    if problem == 'binary':
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=loss,
                  metrics=['accuracy'])
    return model

# ---------- Callbacks ----------
def make_callbacks(fname):
    return [
        tf.keras.callbacks.ModelCheckpoint(fname, save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]

# ---------- Class weights for imbalanced targets ----------
def compute_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}

ethnicity_class_weight = compute_weights(y_ethnicity_train)
age_class_weight = compute_weights(y_age_train)
# gender is roughly balanced; optional:
gender_class_weight = compute_weights(y_gender_train) if len(np.unique(y_gender_train)) > 1 else None

print("Ethnicity class weights:", ethnicity_class_weight)
print("Age class weights:", age_class_weight)
print("Gender class weights (optional):", gender_class_weight)

# ---------- Train gender (binary) ----------
print("\nTraining Gender Model...")
gender_model = build_improved_model(1, problem='binary')
callbacks = make_callbacks('models/gender_model.h5')
gender_model.fit(X_train, y_gender_train,
                 validation_data=(X_test, y_gender_test),
                 batch_size=64, epochs=30,
                 callbacks=callbacks,
                 class_weight=gender_class_weight or None,
                 verbose=2)
gender_model.save('models/gender_model_final.h5')
print("Saved gender_model_final.h5")

# ---------- Train ethnicity (multiclass) ----------
print("\nTraining Ethnicity Model...")
eth_model = build_improved_model(num_classes=5, problem='multiclass')
callbacks = make_callbacks('models/ethnicity_model.h5')
eth_model.fit(X_train, y_ethnicity_train,
              validation_data=(X_test, y_ethnicity_test),
              batch_size=64, epochs=40,
              callbacks=callbacks,
              class_weight=ethnicity_class_weight,
              verbose=2)
eth_model.save('models/ethnicity_model_final.h5')
print("Saved ethnicity_model_final.h5")

# ---------- Train age (multiclass) ----------
print("\nTraining Age Model...")
age_model = build_improved_model(num_classes=4, problem='multiclass')
callbacks = make_callbacks('models/age_model.h5')
age_model.fit(X_train, y_age_train,
              validation_data=(X_test, y_age_test),
              batch_size=64, epochs=40,
              callbacks=callbacks,
              class_weight=age_class_weight,
              verbose=2)
age_model.save('models/age_model_final.h5')
print("Saved age_model_final.h5")

print("\nAll improved models trained & saved.")