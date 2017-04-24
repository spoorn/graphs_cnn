import os
import subprocess
import shutil
from PIL import Image

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
        print filename
        count += 1
      average_ratio += min(ratio, 1.0/ratio)
  print minw, maxw, minh, maxh, minratio, maxratio, minimage
  print count
  print average_ratio/6494

# Resize images
def resize():
  for filename in sorted(os.listdir(os.getcwd()+"/../../data/examples")):
    if filename.endswith(".jpg"):
      im = Image.open(os.getcwd()+"/../../data/examples/"+filename)
      width, height = im.size
      ratio = float(width)/height
      #if ratio <= 4.0 and ratio >= 1.0/4:
      print filename
      im = im.resize((28,28), Image.ANTIALIAS)
      im.save(os.getcwd()+"/../../data/resized_images/"+filename, quality=90, optimize=True)

if __name__ == '__main__':
  #relabel()
  resize()
