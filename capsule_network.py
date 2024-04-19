import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert y_train and y_test to numpy arrays
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)

# Define the Capsule Network model
class Capsule(tf.keras.layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True, activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = self.squash
        else:
            self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel', shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule), \
                                     initializer='glorot_uniform', trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel', shape=(input_num_capsule, input_dim_capsule, \
                                                                  self.num_capsule * self.dim_capsule), \
                                     initializer='glorot_uniform', trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = tf.keras.backend.conv1d(tf.expand_dims(u_vecs, axis=-1), tf.squeeze(self.W, axis=0))
        else:
            u_hat_vecs = tf.keras.backend.local_conv1d(tf.expand_dims(u_vecs, axis=-1), self.W, [1], [1])
        batch_size = tf.shape(u_vecs)[0]
        input_num_capsule = tf.shape(u_vecs)[1]
        u_hat_vecs = tf.reshape(u_hat_vecs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
        u_hat_vecs = tf.transpose(u_hat_vecs, (0, 2, 1, 3))
        b = tf.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            b = tf.transpose(b, (0, 2, 1))
            c = tf.nn.softmax(b)
            c = tf.transpose(c, (0, 2, 1))
            b = tf.transpose(b, (0, 2, 1))
            outputs = self.activation(tf.keras.backend.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b += tf.keras.backend.batch_dot(outputs, u_hat_vecs, [2, 3])
        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def squash(self, x, axis=-1):
        s_squared_norm = tf.keras.backend.sum(tf.keras.backend.square(x), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / tf.keras.backend.sqrt(s_squared_norm + tf.keras.backend.epsilon())
        return scale * x

input_shape = x_train_scaled.shape[1:]

x_input = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Dense(64, activation='relu')(x_input)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = Capsule(num_capsule=10, dim_capsule=16, routings=3)(x)
x = tf.keras.layers.Flatten()(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=x_input, outputs=output)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(x_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)


loss, accuracy = model.evaluate(x_test_scaled, y_test)
print("Test Accuracy:", accuracy)
