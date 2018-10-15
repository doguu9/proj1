import os
cpu_cores = [0, 1, 2, 3]
os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

import matplotlib
matplotlib.use('Agg')
import imutils
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.decomposition import PCA
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import scipy
from scipy import io
import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from perception import CameraIntrinsics, ColorImage, BinaryImage
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

train_dir = '/home/kate/proj1/imgs/single_obj_dataset/train/'
validation_dir = '/home/kate/proj1/imgs/single_obj_dataset/validation/'
image_size = 224
folder = 'vgg19_3'
n_classes = 10
base = "vgg19"
n_layers = 3
ep = 20
learning_rate = 9e-5
notes = "vgg19 9e-5 learning rate"

def init_model(image_size):

    # Load the VGG model
    #vgg_conv = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    base_model = VGG19(weights='imagenet', input_shape=(image_size, image_size, 3))
    vgg_conv = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

    # Freeze all the layers
    for layer in vgg_conv.layers[:-n_layers]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    #model.summary()

    return model

def train_model_simple(model, image_size, train_dir, validation_dir):

    # No Data augmentation 
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Change the batchsize according to your system RAM
    train_batchsize = 100
    val_batchsize = 50

    # Data Generator for Training data
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='categorical')

    # Data Generator for Validation data
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=learning_rate),
                metrics=['acc'])

    # Train the Model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2*train_generator.samples/train_generator.batch_size,
        epochs=ep,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1)

    # Save the Model
    model.save('/home/kate/proj1/results/{}/classifier.h5'.format(folder))
    label2index = validation_generator.class_indices
    with open('/home/kate/proj1/results/{}/label_map.json'.format(folder), 'w') as outfile:
        json.dump(label2index, outfile)

    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('/home/kate/proj1/results/{}/acc_curve.png'.format(folder))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('/home/kate/proj1/results/{}/loss_curve.png'.format(folder))

def show_errors(model, image_size, validation_dir):
    # Create a generator for prediction
    validation_datagen = ImageDataGenerator(rescale=1./255)
    val_batchsize = 50
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='categorical',
            shuffle=False)

    # Get the filenames from the generator
    fnames = validation_generator.filenames

    # Get the ground truth from generator
    ground_truth = validation_generator.classes

    # Get the label to class mapping from the generator
    label2index = validation_generator.class_indices

    # Getting the mapping from class index to class label
    idx2label = dict((v,k) for k,v in label2index.items())

    # Get the predictions from the model using the generator
    predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size,verbose=1)
    predicted_classes = np.argmax(predictions,axis=1)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

    # Show the errors
    error_list = []

    for i in range(len(errors)):
        pred_class = np.argmax(predictions[errors[i]])
        pred_label = idx2label[pred_class]
        
        title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
            fnames[errors[i]].split('/')[0],
            pred_label,
            predictions[errors[i]][pred_class])
        
        original = load_img('{}/{}'.format(validation_dir,fnames[errors[i]]))
        plt.figure(figsize=[7,7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(original)
        plt.show()

        error_list.append([fnames[errors[i]].split('/')[0], pred_label, predictions[errors[i]][pred_class]])

    # DATA: Write list of errors to txt file
    f = open('/home/kate/proj1/results/{}/errors.txt'.format(folder), 'a')
    for entry in error_list:
        f.write('Original label: {}, Prediction: {}, Confidence: {:.3f}\n'.format(entry[0], entry[1], entry[2]))
    f.close()

    # DATA: Write description of params to txt file
    f = open('/home/kate/proj1/results/{}/description.txt'.format(folder), 'a')
    f.write("Base: {}\nTrainable layers: {}\nLearning Rate: {}\nTraining Directory: {}\nValidation Directory: {}\nEpochs: {}\nAdditional Notes: {}".format(base, n_layers, learning_rate, train_dir, validation_dir, ep, notes))
    f.close()

    f = open('/home/kate/proj1/results/{}/raw.txt'.format(folder), 'a')
    l = len(predictions)
    for n in range(l):
        f.write('Label: {}    Predictions: {}\n'.format(idx2label[ground_truth[n]], predictions[n]))
    f.close()

    for index in range(n_classes):
        pr_curve(ground_truth, predictions, idx2label, index)
    pr_all(ground_truth, predictions, idx2label, n_classes)


def pr_curve(ground_truth, predictions, idx2label, index):
    label = []
    confidence = []
    for elem in ground_truth:
        label.append(1) if elem == index else label.append(0)
    for elem in predictions:
        confidence.append(elem[index])
    label = np.asarray(label)
    confidence = np.asarray(confidence)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label, confidence)
    plt.figure()
    plt.plot(recall, precision, 'b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{} Precision Recall Curve'.format(idx2label[index]))
    plt.savefig('/home/kate/proj1/results/{}/pr_curves/pr_{}.png'.format(folder, idx2label[index]))

def pr_all(ground_truth, predictions, idx2label, n_index):
    label = []
    confidence = []
    for i in range(n_index):
        for elem in ground_truth:
            label.append(1) if elem == i else label.append(0)
        for elem in predictions:
            confidence.append(elem[i])
    label = np.asarray(label)
    confidence = np.asarray(confidence)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label, confidence)
    plt.figure()
    plt.plot(recall, precision, 'b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Multilabel Precision Recall Curve')
    plt.savefig('/home/kate/proj1/results/{}/pr_curves/pr_all.png'.format(folder))

if __name__ == '__main__':
    model = init_model(image_size)
    train_model_simple(model, image_size, train_dir, validation_dir)
    show_errors(model, image_size, validation_dir)
