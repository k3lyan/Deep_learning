from keras.models import Sequential
from keras.layers import Dense, Activation
from util import get_normalized_data, T_indicator
import matplotlib.pyplot as plt

X_train, X_test, t_train, t_test = get_normalized_data()
# By default, Keras wants one-hot encoding labels
# there is another cost function we can use,
# where we can directly pass the integer labels straightaway
T_train = T_indicator(t_train)
T_test = T_indicator(t_test)

# Dimensionality
N, D = X_train.shape
K = len(set(t_train))

# The model will be a Sequence of layers
model = Sequential()

# Dense = fully-connected
# ANN with layers dimensionality: D = [784] -> [500] -> [300] -> K = [10]
model.add(Dense(units=500, input_dim=D))
model.add(Activation('relu'))
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(
        loss = 'categorical_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy']
        )

r = model.fit(X_train, T_train, validation_data=(X_test, T_test), epochs=15, batch_size=32)
print('Model fit: {}'.format(r))
print(r.history.keys())

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

