

```python
# % matplotlib
from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
```

    1.13.1



```python
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```


```python
train_images.shape
```




    (60000, 28, 28)




```python
train_labels[0:3]

train_labels
```




    array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)




```python
test_images.shape
```




    (10000, 28, 28)




```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```


![png](basic%20classification_files/basic%20classification_6_0.png)



```python
train_images = train_images / 255.0

test_images = test_images / 255.0
```


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
```


![png](basic%20classification_files/basic%20classification_8_0.png)



```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

    WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/python/ops/resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.



```python
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```


```python
model.fit(train_images, train_labels, epochs=5)
```

    Epoch 1/5
    60000/60000 [==============================] - 13s 212us/sample - loss: 0.4969 - acc: 0.8245
    Epoch 2/5
    60000/60000 [==============================] - 13s 212us/sample - loss: 0.3761 - acc: 0.8648
    Epoch 3/5
    60000/60000 [==============================] - 14s 231us/sample - loss: 0.3369 - acc: 0.8761
    Epoch 4/5
    60000/60000 [==============================] - 15s 250us/sample - loss: 0.3143 - acc: 0.8850
    Epoch 5/5
    60000/60000 [==============================] - 12s 203us/sample - loss: 0.2961 - acc: 0.8914





    <tensorflow.python.keras.callbacks.History at 0x7ff6ebf37f98>




```python
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
```

    10000/10000 [==============================] - 1s 81us/sample - loss: 0.3704 - acc: 0.8663
    Test accuracy: 0.8663



```python
predictions = model.predict(test_images)
```


```python
predictions[0]
```




    array([1.28024376e-05, 2.93391025e-07, 4.49451545e-06, 1.28729571e-05,
           3.31219007e-06, 4.35216073e-03, 1.35796154e-05, 3.72250453e-02,
           1.38637415e-05, 9.58361566e-01], dtype=float32)




```python
np.argmax(predictions[0])
```




    9




```python
test_labels[0]
```




    9




```python
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
```


```python
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
```


![png](basic%20classification_files/basic%20classification_18_0.png)



```python
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()
```


![png](basic%20classification_files/basic%20classification_19_0.png)



```python
# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()
```


![png](basic%20classification_files/basic%20classification_20_0.png)



```python
# Grab an image from the test dataset
img = test_images[0]

print(img.shape)
```

    (28, 28)



```python
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)
```

    (1, 28, 28)



```python
predictions_single = model.predict(img)

print(predictions_single)
```

    [[1.2802449e-05 2.9339049e-07 4.4945155e-06 1.2872945e-05 3.3121901e-06
      4.3521649e-03 1.3579603e-05 3.7225027e-02 1.3863729e-05 9.5836157e-01]]



```python
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
```


![png](basic%20classification_files/basic%20classification_24_0.png)



```python
np.argmax(predictions_single[0])
```




    9




```python

```


```python

```
