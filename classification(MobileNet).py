import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNet
from sklearn.model_selection import train_test_split
import os
import shutil


class PetImageClassifier:
    def __init__(self, input_shape=(128, 128, 3), categories=['Cat', 'Dog'], num_classes=1,
                 model_path='saved_model_mobilenet.h5'):
        self.input_shape = input_shape
        self.categories = categories
        self.num_classes = num_classes
        self.model_path = model_path

        if os.path.exists(self.model_path):
            print(f"Загрузка сохраненной модели из {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
        else:
            print("Создание новой модели MobileNet")
            self.base_model = MobileNet(weights='imagenet', include_top=False, input_shape=self.input_shape)
            self.model = self.build_model()

    def build_model(self):
        self.base_model.trainable = False

        model = models.Sequential()
        model.add(self.base_model)
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(self.num_classes, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, train_generator, validation_generator, epochs=30, steps_per_epoch=50, validation_steps=20):
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps
        )

        print(f"Сохранение модели в {self.model_path}")
        self.model.save(self.model_path)
        return history

    def evaluate(self, test_generator):
        return self.model.evaluate(test_generator)

    def predict(self, img_array):
        return self.model.predict(img_array)


def copy_files(image_list, dest_dir):
    for img_path, category in image_list:
        dest_category_dir = os.path.join(dest_dir, category)
        os.makedirs(dest_category_dir, exist_ok=True)
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(dest_category_dir, img_name))


def train_classifier(data_dir):
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

    copy_files(train_images, 'train')
    copy_files(val_images, 'validation')
    copy_files(demo_images, 'demo')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(128, 128), 
        batch_size=20,
        class_mode='binary')

    validation_generator = val_datagen.flow_from_directory(
        'validation',
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary')

    classifier = PetImageClassifier(categories=categories, model_path='saved_model_mobilenet.h5')

    classifier.train(train_generator, validation_generator)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        'demo',
        target_size=(128, 128),
        batch_size=20,
        class_mode='binary')

    test_loss, test_acc = classifier.evaluate(test_generator)
    print(f'Test accuracy: {test_acc}')

    with open('classification_results.txt', 'w') as file:
        file.write(f'Test accuracy: {test_acc}\n\n')
        file.write('Image Name, Predicted Class\n')

        for i in range(len(test_generator.filenames)):
            img_path = test_generator.filepaths[i]
            img_name = os.path.basename(img_path)

            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)

            prediction = classifier.predict(img_array)

            predicted_class = classifier.categories[1] if prediction[0] > 0.5 else classifier.categories[0]
            file.write(f'{img_name}, {predicted_class}\n')

    print("Results saved to classification_results.txt")


train_classifier('/Users/idg0d/PycharmProjects/tst/PetImages')
