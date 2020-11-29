import os
import sys
import argparse
import numpy as np
import json
# disable tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

configs = dict(
  training_dir = "./data/",
  validation_dir = "./data-test/",
  model_file = "model",
  batch_size = 200,
  epochs = 25,
  target_size = [50, 50],
)

def train():
  print("")
  print("python version", sys.version)
  print("tensorflow version", tf.__version__)
  print("")

  # setup traning data
  training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
  )

  print("Training:")
  train_generator = training_datagen.flow_from_directory(
    configs["training_dir"],
    target_size=(configs["target_size"][0], configs["target_size"][1]),
    class_mode="categorical",
    batch_size=configs["batch_size"]
  )
  print(train_generator)
  print("")


  # write labels to file and log
  labels = train_generator.class_indices
  f = open(configs["model_file"] + ".json", "w")
  f.write(json.dumps(labels))
  f.close()
  print("Class labels")
  print(labels)
  print("")


  # setup validation data
  validation_datagen = ImageDataGenerator(rescale = 1./255)
  print("Validation:")
  validation_generator = validation_datagen.flow_from_directory(
    configs["validation_dir"],
    target_size=(configs["target_size"][0], configs["target_size"][1]),
    class_mode="categorical",
    batch_size=configs["batch_size"]
  )
  print("")


  # define model network
  model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(configs["target_size"][0], configs["target_size"][1], 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation="relu"),
    # number of labels- get it from train_generator
    tf.keras.layers.Dense( len(labels.keys()) , activation="softmax")
  ])
  model.summary()


  # train the model
  model.compile(loss = "categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
  history = model.fit(
    train_generator,
    epochs=configs["epochs"],
    steps_per_epoch=None,
    validation_data=validation_generator,
    verbose=1,
    validation_steps=1
  )
  print("traning done")
  model.save(configs["model_file"] + ".h5")
  print("model saved")

def label_by_value(value):
  with open(configs["model_file"] + ".json") as json_file:
    labels = json.load(json_file)
    label = list(labels.keys())[list(labels.values()).index( value )]
    return label

def labels():
  with open(configs["model_file"] + ".json") as json_file:
    labels = json.load(json_file)
    return list(labels.keys())

def predict(img_src):
  model = tf.keras.models.load_model(configs["model_file"] + ".h5")
  img = image.load_img(img_src, target_size=(configs["target_size"][0], configs["target_size"][1]))
  x = np.expand_dims( image.img_to_array(img) , axis=0)
  images = np.vstack([x])
  classes = model.predict(images, batch_size=1)
  value = classes.argmax(axis=-1)[0]
  label = label_by_value(value)
  return label

def predict_test():
  # list test files
  testdir = configs['validation_dir']
  subdirs = os.listdir(testdir)
  testFiles = []

  for subdir in subdirs:
    subdir = testdir + subdir + "/"
    files = os.listdir(subdir)
    for f in files:
      f = subdir + f
      testFiles.append(f)

  # predict test files
  for testFile in testFiles:
    print(predict(testFile))

# CLI SCRIPT
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train image classification model and predict if image is in forest or city.")
  parser.add_argument("command", nargs="+", help="train, predict, predict_test, labels")
  args = parser.parse_args()
  cmd = args.command[0]
  opts = args.command[1:]

  if(cmd == "train"): train()
  elif(cmd == "predict"): print(predict(opts[0]))
  elif(cmd == "predict_test"): predict_test()
  elif(cmd == "labels"): print(labels())
  else: raise Exception("only supporting command: train, predict, predict_test, labels")
