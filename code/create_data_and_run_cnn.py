from __future__ import print_function

import os
import subprocess
import sys
import shutil

def cleanup():
  print("*** Will not remove .jpg files created from the .ps files ***")
  print("Cleaning up...")
  os.remove(os.getcwd()+'/preprocess_code/data_filenames.csv')
  shutil.rmtree(os.getcwd()+'/preprocess_code/128min_images')
  shutil.rmtree(os.getcwd()+'/preprocess_code/data_csvs')

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], " directory")
  else:
    try:
      print(sys.argv[0] + ": Running scripts to create CNN training data...")
      if not os.path.exists(sys.argv[1]):
        print(sys.argv[1] + " does not exist")
        exit(1)
      outdir = os.path.abspath(sys.argv[1])
      preprocess_wd = 'preprocess_code'
      subprocess.call("python convert_ps2jpg.py " + outdir, shell=True, cwd=preprocess_wd)
      subprocess.call("python resize_images.py " + outdir, shell=True, cwd=preprocess_wd)
      subprocess.call("python image2matrix.py", shell=True, cwd=preprocess_wd)
      print("Running images through CNN...")
      subprocess.call("python load_2layer_spp.py", shell=True)
    except:
      pass
    finally:
      cleanup()
