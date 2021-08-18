import tensorflow as tf
import argparse

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision,Recall
from tensorflow.keras.preprocessing import image
from sklearn import metrics
from configparser import ConfigParser, ExtendedInterpolation

def main(path_to_config):

    main_config = ConfigParser(interpolation=ExtendedInterpolation())
    main_config.read(path_to_config)

    dataset_dir = main_config["DATA"].get('dataset_dir')

    img_height = main_config["DATA"].getint('img_height')
    img_width = main_config["DATA"].getint('img_width')
    batch_size = main_config["DATA"].getint('batch_size')

    # just to try out the parameters
    train_datagen = image.ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True,
                                    validation_split=0.5)

    train_generator = train_datagen.flow_from_directory(dataset_dir, 
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        subset='training') # set as training data

    validation_generator = train_datagen.flow_from_directory(dataset_dir,
                                                            target_size=(img_height, img_width),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            subset='validation') # set as validation data

    #transfer learning 
    base_model = MobileNetV2(input_shape= (img_height, img_width, 3),
                            include_top=False,
                            weights='imagenet')
    # freeze base_model 
    base_model.trainable = False


    base_learning_rate = main_config["TRAIN"].getfloat('base_learning_rate')

    # model architecture
    inputs = Input(shape=(img_height, img_width, 3))
    base = base_model(inputs, training=False)
    pool1 = MaxPooling2D(pool_size=(2, 2))(base)
    flat = Flatten()(pool1)
    hidden_1 = Dense(500, activation = 'relu')(flat)
    hidden_2 = Dense(30, activation='relu')(hidden_1)
    out_layer = Dense(16, activation='softmax')(hidden_2)
    model = Model(inputs=inputs, outputs=out_layer)

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()])

    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator, 
                        epochs = 50, verbose = 2)
    
    model.save(main_config["TRAIN"].get('model_save_path'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config', default='config/cfg.ini', help='Path to config file')
    args = parser.parse_args()
    main(args.path_to_config)
