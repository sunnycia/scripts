# Refer to https://gist.github.com/glombard/7cd166e311992a828675

from __future__ import print_function
import os

from PIL import Image

files = [
  'siti_scatter_plot_matlab/VSG.png',
  'siti_scatter_plot_matlab/SAVAM.png',
  'siti_scatter_plot_matlab/LEDOV.png',
  'siti_scatter_plot_matlab/HOLLYWOOD.png',
  'siti_scatter_plot_matlab/GAZECOM.png',
  'siti_scatter_plot_matlab/DIEM.png',
  'siti_scatter_plot_matlab/DHF1K.png']

width = 1167
height= 875

col=4
row=2
result = Image.new("RGB", (width*col, height*row))

for index, file in enumerate(files):
    path = os.path.expanduser(file)
    print(index, path)
    img = Image.open(path)
    img.thumbnail((width, height), Image.ANTIALIAS)
    x = index % col * width
    y = index // col * height
    w, h = img.size
    print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
    result.paste(img, (x, y, x + w, y + h))

result.save(os.path.expanduser('combine.png'))