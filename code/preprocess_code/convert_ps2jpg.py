import os
import subprocess
import shutil

def relabel():
  count = 0
  for filename in sorted(os.listdir(os.getcwd()+"/../../data/examples")):
    filename = "../../data/examples/" + filename
    newname = "../../data/examples/" + filename[:-3]
    print newname
    #newname = "/../../data/examples/ex" + str(count).zfill(7)
    #print filename
    #print newname
    #command_rename = "mv " + filename + " " + newname + ".ps"
    command_convert = "gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -r300 -sOutputFile="+newname+".jpg "+newname+".ps"
    subprocess.Popen(command_convert, shell=True)
    count += 1

if __name__ == '__main__':
  relabel()
