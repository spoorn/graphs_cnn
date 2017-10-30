from __future__ import print_function

import os
import subprocess
import shutil
import sys
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def relabel():
  minw = None
  maxw = None
  minh = None
  maxh = None
  minratio = None
  maxratio = None
  minimage = ""
  count = 0
  average_ratio = 0.0
  for filename in sorted(os.listdir(os.getcwd()+"/../../data/examples")):
    if filename.endswith(".jpg"):
      #print filename
      im = Image.open(os.getcwd()+"/../../data/examples/"+filename) 
      width, height = im.size
      ratio = float(width)/height
      if minw is None or width < minw:
        minw = width
      if maxw is None or width > maxw:
        maxw = width
      if minh is None or height < minh:
        minh = height
      if maxh is None or height > maxh:
        maxh = height
      if minratio is None or ratio < minratio:
        minratio = float(width)/height
        minimage = filename
      if maxratio is None or ratio > maxratio:
        maxratio = float(width)/height
      if ratio <= 4.0 and ratio >= 1.0/4:
        print(filename)
        count += 1
      average_ratio += min(ratio, 1.0/ratio)
  print(minw, maxw, minh, maxh, minratio, maxratio, minimage)
  print(count)
  print(average_ratio/6494)

# Resize images
def resize(path, output_path, min_dim):
  filenames = []
  for filename in sorted(os.listdir(path)):
    if filename.endswith(".jpg"):
      im = Image.open(path+"/"+filename)
      width, height = im.size
      ratio = float(width)/height
      #if ratio <= 4.0 and ratio >= 1.0/4:
      filenames.append(filename[:-4])
      if min(width, height) > min_dim:
        if width < height:
          im = im.resize((min_dim,height/width*min_dim), Image.ANTIALIAS)
        else:
          im = im.resize((width/height*min_dim,min_dim), Image.ANTIALIAS)
      im.save(os.getcwd()+"/"+output_path+"/"+filename, quality=90, optimize=True)
  write_filenames(filenames)

# Writes filenames to a txt file
def write_filenames(filenames):
  with open('data_filenames.csv', 'w+') as f:
    f.write("\n".join(filenames)) 

if __name__ == '__main__':
  outdir = '128min_images'
  if not os.path.exists(outdir):
    os.mkdir(outdir)
  print(sys.argv[0] + ": Resizing images in " + sys.argv[1] + " and outputting to", os.path.abspath(outdir))
  print("Note: images will have minimum dimension of 128 pixels and preserve aspect ratio")
  resize(sys.argv[1], outdir, 128)
