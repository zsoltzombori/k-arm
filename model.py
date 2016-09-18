from arm import ArmLayer
from keras.layers import Input

image_shape=(100,28,28)
inputs = Input(shape=image_shape)
outputs = ArmLayer(iteration = 20, threshold = 0.05, dict_size=400)(inputs)

#model = Model(input=inputs, output=outputs)
#sgd = SGD(lr=0.1, momentum=0.9, nerestov=True)
#model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])


# print "Average density of nonzero elements in the code: ", np.average(nonzero) / dict_size
# print "Reconstruction error: ", reconsError


# # W = np.load(file("/home/daniel/experiments/k-arm/k-arm/dict.npz"))['arr_0']

    

# from keras.datasets import mnist
# # The data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# nb_features = 28*28
# # flatten the 28x28 images to arrays of length 28*28:
# X_train = X_train.reshape(60000, nb_features)
# X_test = X_test.reshape(10000, nb_features)

# # convert brightness values from bytes to floats between 0 and 1:
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# #print(X_train.shape[0], 'train samples')
# #print(X_test.shape[0], 'test samples')
#     dict_size = 400

# X = X_test[:999]





# def vis(X, filename, n=20):
#     w = 28
#     assert len(X) >= n*n
#     X = X[:n*n]
#     X = X.reshape((n, n, w, w))
#     img = np.zeros((n*w, n*w))
#     for i in range(n):
#         for j in range(n):
#             img[i*w:(i+1)*w, j*w:(j+1)*w] = X[i, j, :, :]
#     img = img.clip(0, 255).astype(np.uint8)
#     scipy.misc.imsave(filename, img)

# vis(Y.dot(W) * 255, "k-arm.png")
