import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

project_dir_name = os.getcwd()
data_path = os.path.join(project_dir_name, 'Datasets')
training_dir = os.path.join(data_path, 'att_faces', 'Training')
testing_dir = os.path.join(data_path, 'att_faces', 'Testing')

imageSize = 92*112
img_width = 92
img_height = 112
batch_size = 5


def load_dataset():
    generator = ImageDataGenerator(rescale=1./255)

    train_data = generator.flow_from_directory(training_dir,
                                               target_size=(
                                                   img_width, img_height,),
                                               batch_size=batch_size,
                                               class_mode='categorical')

    validation_data = generator.flow_from_directory(testing_dir,
                                                    target_size=(
                                                        img_width, img_height,),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
    print(train_data)
    return train_data, validation_data


load_dataset()
