import matplotlib.pyplot as plt
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f"train data shape={train_images.shape}, test data shape={test_images.shape}")
print(f"train label size={len(train_labels)}, test label size={len(test_labels)}")

def plotImage(index):
    plt.title("The image marked as %d" % train_labels[index])
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

def plotTestImage(index):
    plt.title('the image marked as %d' % test_labels[index])
    plt.imshow(test_images[index], cmap='binary')
    plt.show()

plotImage(539)
plotTestImage(1000)
