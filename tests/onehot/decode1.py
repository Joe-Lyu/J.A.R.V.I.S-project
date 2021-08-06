import numpy as np
from keras.utils import to_categorical

data = np.array([1, 5, 3, 8])
print(data)

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

encoded_data = encode(data)
print(encoded_data)

def decode(datum):
    return np.argmax(datum)

# 单个解
for i in range(encoded_data.shape[0]):
    datum = encoded_data[i]
    print('index: %d' % i)
    print('encoded datum: %s' % datum)
    decoded_datum = decode(encoded_data[i])
    print('decoded datum: %s' % decoded_datum)
    print()

# 全部解
print(np.argmax(encoded_data, axis=1))