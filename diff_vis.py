import numpy as np
import PIL.Image as Image

def get_numpy_picture_array(X, n_x, n_y):
    height, width = X.shape[-2:]
    image_data = np.zeros(
        ((height+1) * n_y - 1, (width+1) * n_x - 1),
        dtype='uint8'
    )
    n = len(X)
    assert n <= n_x * n_y
    for idx in xrange(n):
        x = idx % n_x
        y = idx / n_x
        sample = X[idx]
        image_data[(height+1)*y:(height+1)*y+height, (width+1)*x:(width+1)*x+width] = (255*sample).clip(0, 255)
    return image_data

# That's awkward, (height, width) corresponds to (n_y, n_x),
# namely the image size is ~ (width*n_x, height*n_y), but the order is reversed between the two.
def diff_vis(dataOriginal, generatedOriginal, height, width, n_x, n_y, name, distances=None):
    data = dataOriginal.copy().reshape((-1, height, width))
    generated = generatedOriginal.copy().reshape((-1, height, width))
    if distances is not None:
        # Ad hoc values, especially now that there's no bimodality aka left bump of 1s.
        VALLEY = 0.3
        MAX_OF_REASONABLE = 1.0
        assert len(distances) == len(data) == len(generated)
        for i,distance in enumerate(distances):
            length = np.linalg.norm(data[i])
            relativeDistance = (distance+1e-5)/(length+1e-5)
            barHeight = min((int(height*relativeDistance/MAX_OF_REASONABLE), height))
            goGreen = float(relativeDistance<VALLEY) # 1.0 if we want green, 0.0 if we want red.
            # data is drawn red, generated is drawn green,
            # we hackishly manipulate them here to get the needed color.
            data     [i, :barHeight, :2] = 1.0-goGreen
            generated[i, :barHeight, :2] = goGreen

    image_data      = get_numpy_picture_array(data, n_x, n_y)
    image_generated = get_numpy_picture_array(generated, n_x, n_y)
    # To color-combine the images AFTER they are arranged in a grid is
    # more than a little hackish.
    blue = np.minimum(image_data, image_generated) # image_data/2 + image_generated/2
    rgb = np.dstack((image_data, image_generated, blue))
    img = Image.fromarray(rgb, 'RGB')
    img.save(name)
