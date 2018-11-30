import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(net, layer_name, padding=0, filename='', visualize=False):
    # refer to 
    #     https://www.eriksmistad.no/visualizing-learned-features-of-a-caffe-neural-network/
    
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[2]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size):
                for j in range(filter_size):
                    result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j] = data[n, c, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
    # print result.shape;exit()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')
    # Plot figure
    if visualize:
        plt.show()
    else:
        result = result * 255
        cv2.imwrite(filename, result)
        # Save plot if filename is set
        # if filename != '':
        #     plt.savefig(filename)
            # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

def visualize_3dweights(net, layer_name, padding=0, filename='', visualize=False):
    # refer to 
    #     https://www.eriksmistad.no/visualizing-learned-features-of-a-caffe-neural-network/
    
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0]*data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    assert data.shape[3]==data.shape[4]
    filter_size = data.shape[3]
    # Size of the result image including padding
    result_size = filters_per_row*(filter_size + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size, result_size, data.shape[2]))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    # filter_z = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for d in range(data.shape[2]):
                for i in range(filter_size):
                    for j in range(filter_size):
                        result[filter_y*(filter_size + padding) + i, filter_x*(filter_size + padding) + j, d] = data[n, c, d, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)
    # print result.shape;exit()
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='gray', interpolation='nearest')
    # Plot figure
    if visualize:
        plt.show()
    else:
        result = result * 255
        cv2.imwrite(filename, result)
        # Save plot if filename is set
        # if filename != '':
        #     plt.savefig(filename)
            # plt.savefig(filename, bbox_inches='tight', pad_inches=0)

