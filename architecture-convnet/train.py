from pathlib import Path
from collections import namedtuple

import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import (Conv2D, MaxPooling2D, Dense, Activation, Flatten,
                          Dropout)

MODEL_PATH = Path('model.h5')

MAX_EXAMPLES = float('inf')

EXAMPLES = Path('examples')

SIZE = 128

CHANNELS = 3

class LabeledExamples(namedtuple('LabeledExamples', ['examples', 'labels'])):
    @classmethod
    def create_binary(cls, a, b):
        a_labels = np.zeros((a.shape[0], 1))
        b_labels = np.ones((b.shape[0], 1))

        return cls(
            examples=np.concatenate([a, b]),
            labels=np.concatenate([a_labels, b_labels])
        )

    @property
    def num_examples(self):
        return self.examples.shape[0]

    def shuffled(self):
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)

        return self.__class__(
            examples=self.examples[indices],
            labels=self.labels[indices]
        )

    def split(self, start, end):
        return self.__class__(
            examples=self.examples[start:end],
            labels=self.labels[start:end],
        )

    def split_into_train_dev_test(self, train_pct=0.60, dev_pct=0.20):
        train_end_index = int(self.num_examples * train_pct)
        dev_end_index = train_end_index + int(self.num_examples * dev_pct)

        return (
            self.split(0, train_end_index),
            self.split(train_end_index, dev_end_index),
            self.split(dev_end_index, self.num_examples)
        )


def load_examples(directory):
    filenames = list(directory.glob('*.jpg'))
    m = min(len(filenames), MAX_EXAMPLES)
    examples = np.zeros((m, SIZE, SIZE, CHANNELS), dtype=np.float32)
    for ex, i in zip(filenames, range(m)):
        img = Image.open(str(ex))
        img = img.resize((SIZE, SIZE))
        npimg = np.fromstring(img.tobytes(), dtype=np.uint8)
        examples[i] = npimg.reshape((SIZE, SIZE, CHANNELS)) / 255.0
    return examples


np.random.seed(1)

train, dev, test = LabeledExamples.create_binary(
    load_examples(EXAMPLES / 'art-deco'),
    load_examples(EXAMPLES / 'beaux-arts')
).shuffled().split_into_train_dev_test()

def create_model():
    # VGG-like convnet from:
    # https://keras.io/getting-started/sequential-model-guide/
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
              input_shape=(SIZE, SIZE, CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


if MODEL_PATH.exists():
    print(f"Loading model from {MODEL_PATH}.")
    model = load_model(str(MODEL_PATH))
else:
    model = create_model()
    model.fit(train.examples, train.labels, batch_size=32, epochs=30)
    print(f"Saving model to {MODEL_PATH}.")
    model.save(str(MODEL_PATH))

score = model.evaluate(test.examples, test.labels, batch_size=32)

print(model.metrics_names)
print(score)
