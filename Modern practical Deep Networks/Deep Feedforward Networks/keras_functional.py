from keras.models import Model
from keras.layers import Dense, Input
from util import get_normalized_data, T_indicator

import matplotlib.pyplot as plt
X_train, X_test, t_train, t_test = get_normalized_data()

# get shapes
N, D = X_train.shape
K = len(set(t_train))

T_train = T_indicator(t_train)
T_test = T_indicator(t_test)

# ANN with layers [784] -> [500] -> [300] -> [10]
x = Input(shape=(D,))
a = Dense(500, activation='relu')(x)
a = Dense(300, activation='relu')(a)
a = Dense(K, activation='softmax')(a)

model = Model(inputs=x, outputs=a)

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# note: multiple ways to choose a backend
# either theano, tensorflow, or cntk
# https://keras.io/backend/

r = model.fit(X_train, T_train, validation_data=(X_test, T_test), epochs=15, batch_size=32)
print("Returned:", r)

# should see: dict_keys(['val_loss', 'acc', 'loss', 'val_acc'])
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

