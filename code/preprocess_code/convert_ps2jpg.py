from __future__ import print_function

import os
import subprocess
import shutil
import sys

# NOTE: when script is finished, it will hold.  Ctrl-C to finish.

def convertps2jpg(path, outpath):
  count = 0
  FNULL = open(os.devnull, 'w')
  for filename in sorted(os.listdir(path)):
    if filename.endswith(".ps") and not os.path.exists(path+filename[:-3]+".jpg"):
      newname = outpath + "/" + filename[:-3]
      filename = path + "/" + filename
      command_convert = "gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -r300 -sOutputFile="+newname+".jpg "+filename
      subprocess.call(command_convert, shell=True, stdout=FNULL)
      count += 1

if __name__ == '__main__':
  print(sys.argv[0] + ": Converting .ps to .jpg on files in directory", sys.argv[1])
  convertps2jpg(sys.argv[1], sys.argv[1])
