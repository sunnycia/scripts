import os
import cPickle as pkl
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot_dict_path', required=True)
parser.add_argument('--x_length', default=20)
args = parser.parse_args()

plot_dict_path = args.plot_dict_path
if not os.path.isfile(plot_dict_path):
    print plot_dict_path, "not exists!"
plot_dict= pkl.load(open(plot_dict_path, 'rb'))
plot_xlength=args.x_length
plt.subplot(4, 1, 1)
plt.plot(plot_dict['x'][-plot_xlength:], plot_dict['y_loss'][-plot_xlength:])
plt.ylabel('loss')
plt.subplot(4, 1, 2)
plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_cc'][-plot_xlength:])
plt.ylabel('cc metric')
plt.subplot(4, 1, 3)
plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_sim'][-plot_xlength:])
plt.ylabel('sim metric')
plt.subplot(4, 1, 4)
plt.plot(plot_dict['x_valid'][-plot_xlength:], plot_dict['y_kld'][-plot_xlength:])
plt.xlabel('iter')
plt.ylabel('kld metric')

plt.show()