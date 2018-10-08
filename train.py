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

from perception import CameraIntrinsics, ColorImage, BinaryImage

train_dir = '/home/kate/proj1/imgs/training_imgs'
validation_dir = '/home/kate/proj1/imgs/validation_imgs'
image_size = 224
folder = '1'
n_classes = 10
base = "vgg16"
n_layers = 4
ep = 20

def init_model(image_size):

    # Load the VGG model
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

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
    train_batchsize = 70
    val_batchsize = 30

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
                optimizer=optimizers.RMSprop(lr=1e-4),
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

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    # DATA: Write loss and accuracy curves to mat file
    loss_arr = []
    accuracy_arr = []
    for i in range(len(acc)):
        loss_arr.append([loss[i], val_loss[i]])
        accuracy_arr.append([acc[i], val_acc[i]])
    loss_arr = np.asarray(loss_arr)
    accuracy_arr = np.asarray(accuracy_arr)
    scipy.io.savemat('/home/kate/proj1/results/{}/loss.mat'.format(folder), mdict={'loss': loss_arr})
    scipy.io.savemat('/home/kate/proj1/results/{}/accuracy.mat'.format(folder), mdict={'accuracy': accuracy_arr})

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
    predictions = model.predict_generator(validation_generator, steps=validation_generator.samples/validation_generator.batch_size + 1,verbose=1)
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
    f.write("Base: {}\n Trainable layers: {}\n Training Directory: {}\n Validation Directory: {}\n Epochs: {}".format(base, n_layers, train_dir, validation_dir, ep))
    f.close()

    # DATA: Write precision recall curve to mat file
    # Subtask: Create confusion matrix for classification
    confusion_arr = []
    for i in range(0, n_classes):
        new_row = []
        for j in range(0, n_classes):
            new_row.append(0)
        confusion_arr.append(new_row)
    for entry in range(len(ground_truth)):
        confusion_arr[ground_truth[entry]][predicted_classes[entry]] += 1
    # Subtask: Extract Tp, Fp, and Fn values from matrix
    values = []
    subvalues = []
    for entry in range(0, n_classes):
        subvalues.append(0)
    for row in confusion_arr:
        for elem in row:
            subvalues[elem] += 1
    for entry in range(0, n_classes):
        tp = confusion_arr[entry][entry]
        fp = sum(confusion_arr[entry]) - tp
        fn = subvalues[entry]
        values.append([entry, tp, fp, fn])
    #Subtask: Calculate precision and recall values for all classes
    pr_vals = []
    for label in values:
        if label[1] + label[2] != 0:
            precision = float(label[1]) / float(label[1] + label[2])
        else:
            precision = 0
        if label[1] + label[3] != 0:
            recall = float(label[1]) / float(label[1] + label[3])
        else:
            recall = 0
        pr_vals.append([precision, recall])
    #Subtask: Write pr_vals array to mat file
    pr_curve = np.asarray(pr_vals)
    scipy.io.savemat('/home/kate/proj1/results/{}/prcurve.mat'.format(folder), mdict={'pr_curve': pr_curve})

if __name__ == '__main__':
    model = init_model(image_size)
    train_model_simple(model, image_size, train_dir, validation_dir)
    show_errors(model, image_size, validation_dir)
