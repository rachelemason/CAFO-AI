from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import ee


def write_to_file(fc, filename, folder):

  task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=filename,
        folder=folder,
        fileFormat='GeoJSON',
    )

  task.start()

def ee_task_status(n_tasks=5):
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

def get_predictions(model, X_test, y_test, metadata):

  # Create a copy of the test data to ensure it doesn't get altered by preprocess_input
  test_processed = np.copy(X_test)

  # Process the test data in the same way as the training and val data
  test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
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

  def show_images(title):
    plt.figure(figsize=(9, 7))
    for i, idx in enumerate(df2.index[:24]):
      plt.subplot(4, 6, i+1)
      img = X_test[idx].reshape(64, 64, 3)
      img = (img / np.max(img)) * 255
      plt.imshow(img.astype(int))
      plt.axis('off')
      model_class = class_mapping[df2.loc[idx, 'Model Class']]
      prob = df2.loc[idx, 'Max prob']
      plt.title(f"{model_class}: {prob:.2f}",\
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
    df2 = get_probs(df2)
    print(f"{len(df2)} {name} images were correctly classified")
    show_images(title=f"Label={name}, Prediction={name}")

  # Incorrectly-classified images, all categories
  for category, name in class_mapping.items():
    df2 = df[(df.loc[:, 'Label'] == category) &\
             (df.loc[:, 'Label'] != df.loc[:, 'Model Class'])].copy()
    df2 = get_probs(df2)
    print(f"{len(df2)} {name} images were incorrectly classified")
    show_images(title=f"Label={name}, Prediction=other")


def probability_hist(df):

  _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 7))

  df2 = df[(df['Prediction'] == 1) & (df['Label'] == 1)]
  ax1.hist(df2['Probability'], bins=50, density=True, stacked=True)
  ax1.set_xlim(0.5, 1)
  ax1.set_ylim(0, 50)
  ax1.set_title('Label=1, Prediction=1', fontsize=9)

  df2 = df[(df['Prediction'] == 1) & (df['Label'] == 0)]
  ax2.hist(df2['Probability'], bins=50, density=True, stacked=True)
  ax2.set_xlim(0.5, 1)
  ax2.set_ylim(0, 50)
  ax2.set_title('Label=0, Prediction=1', fontsize=9)

  df2 = df[(df['Prediction'] == 0) & (df['Label'] == 1)]
  ax3.hist(df2['Probability'], bins=50, density=True, stacked=True)
  ax3.set_xlim(0.5, 1)
  ax3.set_ylim(0, 50)
  ax3.set_title('Label=1, Prediction=0', fontsize=9)

  df2 = df[(df['Prediction'] == 0) & (df['Label'] == 0)]
  ax4.hist(df2['Probability'], bins=50, density=True, stacked=True)
  ax4.set_xlim(0.5, 1)
  ax4.set_ylim(0, 50)
  ax4.set_title('Label=0, Prediction=0', fontsize=9)

  ax3.set_xlabel("CAFO probability")
  ax3.set_ylabel("Frequency")
  
  plt.tight_layout()
