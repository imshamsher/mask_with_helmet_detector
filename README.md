## Detection model

To identify head/face of person with safety helmet and mask we train detection model for two classes.
Two classes are
- both_helmet_and_mask
- not_both_helmet_and_mask

First step is parse the data from annotation.xml file to yolo format. Pick only head boxes and split it into two classes.

We can use yolo-light from https://github.com/AlexeyAB/yolo2_light

The average precision for tiny yolo on coco dataset is 45 % and average fps 60

# Another approach is also possible

- train face detector
- train mask classifier
- train helmet-no-helmet classifier

Design of pipeline

- load image using opencv
- infer image with face detector
- run mask classifier over cropped faces and store results.
- run helmet classifier over cropped face and store results.

We can use haarcascade filters based face detector which very light

And we can train a sequential model for classification problem

```text
model = tf.keras.models.Sequential([
      lstm_layer,
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(output_size, activation='softmax')])

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])

```

so the model is also very light weight which is use to just classify mask and no mask

Similarly we can train classifier for helmet no helmet.