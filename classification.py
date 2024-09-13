import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import shutil


data_dir = '/Users/idg0d/PycharmProjects/tst/PetImages'  # Убедитесь, что путь к данным правильный
categories = ['Cat', 'Dog']

images = []
for category in categories:
    category_dir = os.path.join(data_dir, category)
    for img_name in os.listdir(category_dir):
        img_path = os.path.join(category_dir, img_name)
        if os.path.isfile(img_path):
            images.append((img_path, category))


train_val_images, demo_images = train_test_split(images, test_size=0.10, random_state=42)
train_images, val_images = train_test_split(train_val_images, test_size=0.22, random_state=42)


def copy_files(image_list, dest_dir):
    for img_path, category in image_list:
        dest_category_dir = os.path.join(dest_dir, category)
        os.makedirs(dest_category_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest_category_dir, img_name))


copy_files(train_images, 'train')
copy_files(val_images, 'validation')
copy_files(demo_images, 'demo')

model = models.Sequential()
model.add(layers.Input(shape=(150, 150, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = val_datagen.flow_from_directory(
    'validation',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    'demo',
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')


with open('classification_results.txt', 'w') as file:
    file.write(f'Test accuracy: {test_acc}\n\n')
    file.write('Image Name, Predicted Class\n')

    for i in range(len(test_generator.filenames)):
        img_path = test_generator.filepaths[i]
        img_name = os.path.basename(img_path)


        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)


        predicted_class = 'Dog' if prediction[0] > 0.5 else 'Cat'

        file.write(f'{img_name}, {predicted_class}\n')

print("Results saved to classification_results.txt")
