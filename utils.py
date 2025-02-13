from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras.applications.efficientnet import preprocess_input as preprocess_input_efficientnet
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
import ee


def write_to_file(fc, filename, folder):
  """
  Write an Earth Engine featureCollection to a Google Drive file.
  """

  task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=filename,
        folder=folder,
        fileFormat='GeoJSON',
    )

  task.start()


def ee_task_status(n_tasks=5):
  """
  Print the status of the last <n_tasks> Earth Engine tasks (e.g. writing to
  file)
  """

  tasks = ee.batch.Task.list()

  # Print the tasks along with their status
  for task in tasks[:n_tasks]:
      status = task.status()
      if status['state'] in ['READY', 'RUNNING', 'COMPLETED']:
        ms = status['start_timestamp_ms']
        print(f"Task {status['id']} started at {datetime.fromtimestamp(ms/1000.0)}")
        print(f"Current status: {status['state']}")
      elif status['state'] == 'FAILED':
          print(f"Task {status['id']} FAILED")
          print("   Error Message:", status['error_message'])
      else:
          print(status)


def get_predictions(model, X_test, y_test, preprocessing, metadata):
  """
  Return a df of model predictions for test data, including associated
  metadata
  """

  # Create a copy of the test data to ensure it doesn't get altered by preprocess_input
  test_processed = np.copy(X_test)

  # Process the test data in the same way as the training and val data
  if preprocessing == "VGG16":
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_vgg16)
  elif "EfficientNet" in preprocessing:
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_efficientnet)
  elif preprocessing == "ResNet50":
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_resnet)
  else:
    raise ValueError("Preprocessing function must be one of\
     VGG16|EfficientNetXX|ResNet50")

  test_generator = test_datagen.flow(test_processed, batch_size=32, shuffle=False)

  # Use model to obtain predictions on rescaled, mean-subtracted test dataset
  y_pred = model.predict(test_generator)

  # Create df with ground-truth label, model probabilities, and model class,
  # where class is the one with the highest probability
  df = pd.DataFrame(columns=['Label', 'Model Probabilities', 'Model Class'])
  for idx, (label, prediction) in enumerate(zip(y_test, y_pred)):
      df.loc[idx, 'Label'] = np.argmax(label) 
      df.loc[idx, 'Model Probabilities'] = y_pred[idx]
      df.loc[idx, 'Model Class'] = np.argmax(prediction)

  # Add metadata
  for column in metadata.columns:
      df.loc[:, column] = metadata[column]

  return df


def plot_classified_images(X_test, df, class_mapping, ascending):
  """
  Make a set of plots showing images that have been correctly and incorrectly
  classified, separated into CAFO and not-CAFO classes.
  """

  def show_images(title):
    plt.figure(figsize=(9, 56))
    for i, idx in enumerate(df2.index[:192]):
      plt.subplot(32, 6, i+1)
      img = X_test[idx].reshape(64, 64, 3)
      img = (img / np.max(img)) * 255
      plt.imshow(img.astype(int))
      plt.axis('off')
      model_class = class_mapping[df2.loc[idx, 'Model Class']]
      prob = df2.loc[idx, 'Max prob']
      dataset = df2.loc[idx, 'Dataset name']
      plt.title(f"{model_class}: {prob:.2f} ({dataset[:3]})",\
                fontsize=9)
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()

  def get_probs(df2):
    for idx, row in df2.iterrows():
      prob = np.max(row['Model Probabilities'])
      df2.loc[idx, 'Max prob'] = prob
    df2.sort_values(by=['Max prob'], ascending=ascending, inplace=True)
    return df2

  # Correctly-classified images, all categories
  for category, name in class_mapping.items():
    df2 = df[(df.loc[:, 'Label'] == category) &\
             (df.loc[:, 'Label'] == df.loc[:, 'Model Class'])].copy()
    if len(df2) > 0:
      df2 = get_probs(df2)
      print(f"{len(df2)} {name} images were correctly classified")
      show_images(title=f"Label={name}, Prediction={name}")
    else:
      print(f"0 {name} images were correctly classified")

  # Incorrectly-classified images, all categories
  for category, name in class_mapping.items():
    df2 = df[(df.loc[:, 'Label'] == category) &\
             (df.loc[:, 'Label'] != df.loc[:, 'Model Class'])].copy()
    if len(df2) > 0:
      df2 = get_probs(df2)
      print(f"{len(df2)} {name} images were incorrectly classified")
      show_images(title=f"Label={name}, Prediction=other")
    else:
      print(f"0 {name} images were incorrectly classified")

def show_random_images(dfs, label, min_prob, fname, rand=42):
  """
  Given a results dataset, create a 5x4 mosaic of randomly-selected
  images with label=label and CAFO probability > min_prob
  """

  df_list = []
  count = 0
  for df in dfs:
    df = df[(df["Probability_0"] > min_prob) & (df["Label"] == label)]
    count += len(df)
    # Sample 2 images if possible, then try 1, then just move on if no
    # images meet the criteria
    try:
      df = df.sample(2, random_state=rand)
    except ValueError:
      try:
        df = df.sample(1, random_state=rand)
      except ValueError:
        continue
    df_list.append(df)

  df = pd.concat(df_list)
  print(f"{count} images have label = {label} and p(CAFO) > {min_prob}")

  _, axes = plt.subplots(5, 4, figsize=(5, 7))
  for ax, img, loc in zip(axes.flatten(), df["Sentinel"], df['Dataset name']):
      img = img.reshape(64, 64, 3)
      img = (img / np.max(img)) * 255
      ax.imshow(img.astype(int))
      ax.set_title(loc, fontsize=8)
  # Make things look nice even if there aren't enough images to fill the
  # subplots
  for ax in axes.flatten():
     ax.axis('off')

  plt.savefig(f'/content/drive/MyDrive/CAFO_data/Analysis/{fname}.png',\
  dpi=450)


def probability_hist(df, ymax, fname):
  """
  Create histograms showing model probabilities CAFOs and not-CAFO
  classes.
  """

  f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(6, 5))

  bins = np.linspace(0, 1.0, 100)
  if 'Probability_0' not in df.columns:
    df['Probability_0'] = df['Model Probabilities'].apply(lambda x: x[0])
    df['Probability_1'] = df['Model Probabilities'].apply(lambda x: x[1])

  # Separate dfs for CAFO and not-CAFO images
  df0 = df[df['Label'] == 0]
  df1 = df[df['Label'] == 1]

  # Histograms

  ax0a = ax0.twinx()
  ax0.hist(df0['Probability_0'], bins=bins, histtype='step', color='0.3')
  ax0a.hist(df0['Probability_0'], bins=bins, histtype='step', color='0.7',\
  cumulative=True, density=True, ls=":")
 
  ax0.set_title(f'{len(df0)} CAFO images', fontsize=8)
  ax0.axhline(y=320, xmin=0.05, xmax=0.2, color='0.3', ls='-', lw=1)
  ax0.text(0.24, 0.95, 'Frequency', va='top', ha='left',\
           transform=ax0.transAxes, fontsize=8)
  ax0.axhline(y=290, xmin=0.05, xmax=0.2, color='0.7', ls=':')
  ax0.text(0.24, 0.85, 'Cumulative frequency', va='top', ha='left',\
           transform=ax0.transAxes, fontsize=8)
  ax0.set_xlabel("CAFO Probability", fontsize=8)

  ax1a = ax1.twinx()
  ax1.hist(df1['Probability_1'], bins=bins, histtype='step', color='0.3')
  ax1a.hist(df1['Probability_1'], bins=bins, histtype='step', color='0.7',\
  cumulative=True, density=True, ls=":")
  ax1.set_title(f'{len(df1)} not-CAFO images', fontsize=8)
  ax1.set_xlabel("not-CAFO Probability", fontsize=8)

  for ax in (ax0, ax1):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', labelsize=8)
    ax.tick_params(axis='y', rotation=90)
    ax.set_ylabel("Frequency", fontsize=8)

  for ax in (ax0a, ax1a):
    ax.set_ylabel("Cumulative frequency", fontsize=8)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', rotation=90, labelsize=8)


  # Aspect-area

  cmap = 'RdYlBu_r'
  pts = ax2.scatter(df0["Area (sq m)"], df0["Aspect ratio"],\
                    c=df0['Probability_0'], s=3, cmap=cmap)
  cax = f.add_axes([0.45, 0.11, 0.015, 0.336])
  c1 = f.colorbar(pts, cax=cax)

  pts = ax3.scatter(df1["Area (sq m)"], df1["Aspect ratio"],\
                    c=df1['Probability_1'], s=3, cmap=cmap)
  cax = f.add_axes([0.908, 0.11, 0.015, 0.336])
  c2 = f.colorbar(pts, cax=cax)

  for cbar in (c1, c2):
    cbar.ax.tick_params(labelsize=8, labelrotation=90)

  for ax in (ax2, ax3):
    ax.tick_params(axis='both', labelsize=8)
    ax.tick_params(axis='y', rotation=90, labelsize=8)
    ax.set_xlim(700, 5999)
    ax.set_ylim(0.8, 20)
    ax.set_yticks([5, 10, 15, 20])
    ax.set_xlabel("Area (sq m)", fontsize=8)
    ax.set_ylabel("Aspect ratio", fontsize=8)
  
  plt.subplots_adjust(wspace=0.45, hspace=0.3)
  plt.savefig(f'/content/drive/MyDrive/CAFO_data/Analysis/{fname}.png',\
  dpi=450)


def select_test_image(test_data, model, results_df, prediction, label,\
                      probability_range=(0.99, 1.0)):
  """
  Return a randomly-selected image for which the label and model prediction
  match the stated criteria, and a version preprocessed for the relevant model.
  """
  
  # Some prep work
  df = pd.DataFrame()
  df['Probability_0'] = results_df['Model Probabilities'].apply(lambda x: x[0])
  df['Probability_1'] = results_df['Model Probabilities'].apply(lambda x: x[1])
  df['Label'] = results_df['Label']
  df['Dataset'] = results_df['Dataset name']

  # Select an image matching criteria
  df = df[df["Label"] == label]
  if prediction == label:
    df = df[df["Label"] == prediction]
  else:
    df = df[df["Label"] != prediction]

  df = df[(df[f'Probability_{str(prediction)}'] > probability_range[0]) &\
          (df[f'Probability_{str(prediction)}'] <= probability_range[1])]

  df = df.sample(1)
  print(f"Dataset: {df['Dataset'].values[0]}")
  print(f"Probability = {df[f'Probability_{str(prediction)}'].values[0]:.2f}")
  test_img = test_data[df.index.values[0]]

  # Create a version that is scaled for imshow
  img = ((test_img / np.max(test_img)) * 255).astype("uint8")

  # Create a version that is preprocessed in the same way as the training data
  for_processing = test_img.copy()
  if model == "VGG16":
    preprocessed_image = preprocess_input_vgg16(for_processing)
  elif "EfficientNet" in model:
    preprocessed_image = preprocess_input_efficientnet(for_processing)
  elif model == "ResNet50":
    preprocessed_image = preprocess_input_resnet(for_processing)
  else:
    raise ValueError("Model must be one of VGG16|EfficientNetXX|ResNet50")
  
  # Add batch dimension to the processed image
  preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  

  return img, preprocessed_image


def get_gradcam_heatmap(model, image, last_conv_layer_name, class_index):
  """
  Given a trained model and a preprocessed image output by select_test_image,
  return the gradient heatmap for the specified image class/category/label
  and concolutional layer
  """

  # Create a model that outputs the feature map and predictions
  grad_model = Model([model.inputs],
      [model.get_layer(last_conv_layer_name).output, model.output])

  # Record operations for automatic differentiation
  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(image)  # Forward pass
      loss = predictions[:, class_index]  # Get the prediction for the target class

  # Calculate gradients with respect to the chosen class
  grads = tape.gradient(loss, conv_outputs)

  # Calculate the mean intensity of the gradients over the width and height
  # This gives the "importance" of each feature map channel
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

  # Multiply each channel in the feature map array by its importance
  conv_outputs = conv_outputs[0]  # Remove the batch dimension
  heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

  # Apply ReLU to the heatmap (only keep positive values)
  heatmap = np.maximum(heatmap, 0)

  # Normalize the heatmap to range [0, 1]
  heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

  return heatmap


def display_gradcam(image, heatmap, alpha=0.5):
  """
  Display a test image from select_test_image, and a gradient heatmap from
  get_gradcam_heatmap.
  """

  # Resize heatmap to match the original image size
  heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

  # Convert the heatmap to an RGB color map
  heatmap_colored = plt.get_cmap("jet")(heatmap_resized)

  # Display the original image
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 2, 1)
  plt.imshow(image)
  plt.axis("off")

  # Overlay the Grad-CAM heatmap
  plt.subplot(2, 2, 2)
  plt.imshow(image)
  plt.imshow(heatmap_colored, alpha=alpha)
  plt.axis("off")

  # Same with a contrast-adjusted image
  plt.subplot(2, 2, 3)
  adjusted = cv2.convertScaleAbs(image, alpha=3, beta=1)
  plt.imshow(adjusted)
  plt.axis("off")

  plt.subplot(2, 2, 4)
  plt.imshow(adjusted)
  plt.imshow(heatmap_colored, alpha=alpha)
  plt.axis("off")

  plt.tight_layout()

