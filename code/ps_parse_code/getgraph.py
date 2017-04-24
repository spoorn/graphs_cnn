#!/usr/bin/python

import os
import sys
import subprocess
import argparse

from epstrim import *
from epsinterpreter import *
from graph_guess import *

parser = argparse.ArgumentParser()
parser.add_argument('-a','--author',action='store',nargs=1,default='',help='author information for database entries')
parser.add_argument('-r','--references',action='store',nargs=1,default='',help="reference information for database entry")
parser.add_argument('-c','--comments',action='store',nargs=1,default='',help="comments for database entry")
parser.add_argument('-t','--title',action='store',nargs=1,default='',help="title for database entry")
parser.add_argument('-n','--name',action='store',nargs=1,default='salt',help="database entry name")
parser.add_argument('-l','--links',action='store',nargs=1,default='',help="links for database entry")
parser.add_argument('-o','--output',action='store',nargs=1,default='png',help='filetype for image output, supports png, jpeg, and eps')
parser.add_argument('eps')

args = parser.parse_args()

##
# convert eps image to graphs and output in json format
##

def write_json(label,metadata,g,ofile):
  ofile.write("\"%s\": {\n" % label)
  ofile.write("  \"name\": \"%s\",\n" % label)
  ofile.write("  \"title\": \"%s\",\n" % args.title[0])
  ofile.write("  \"vertices\": [")
  ofile.write(','.join([str(vert + 1) for vert in g.v]))
  ofile.write("],\n")
  ofile.write("  \"edges\": [")
  ofile.write(','.join(["[%i,%i]" % (edge[0]+1,edge[1]+1) for edge in g.e]))
  ofile.write("],\n")
  ofile.write("  \"embedding\": [")
  ofile.write(','.join(["[%.3f,%.3f]" % (g.x[i],g.y[i]) for i in range(g.n)]))
  ofile.write("],\n")
  ofile.write("  \"degrees\": [")
  degrees = g.get_degree_sequence()
  ofile.write(','.join([str(deg) for deg in degrees]))
  ofile.write("],\n")
  for pair in metadata[:-2]:
    ofile.write("  \"%s\": [\"%s\"],\n" % (pair[0],pair[1]))
  pair = metadata[-1]
  ofile.write("  \"%s\": [\"%s\"]\n" % (pair[0],pair[1]))
  ofile.write("}\n\n")


def extract_graphs(eps_objects,folder='.'):
  graph = graph_guess(eps_objects)
  graphs = get_connected_embedded_subgraphs(graph)

  print("Found %i graphs!" % (len(graphs)))
  nbiggraphs = 0
  for g in graphs:
    if(g.n > 3):  # ignore graphs with less than four vertices
      nbiggraphs += 1
      g.plot()
  
  if(nbiggraphs > 0):
    plt.savefig("%s/%s.png" % (folder,args.name[0]))
    plt.clf()
    ofile_name = "%s/%s.json" % (folder,args.name[0])
    ofile = open(ofile_name,"w")

    metadata = [["comments", args.comments[0]],\
                ["references", args.references[0]],\
                ["links", args.links[0]],\
                ["authors", args.author[0]]]

    for ig in range(len(graphs)):
      g = graphs[ig]
      label = "%s_%s_graph%i" % (args.name[0],args.title[0],ig)
      if(g.n > 3):
        write_json(label,metadata,g,ofile)
    
    ofile.close()


if __name__ == '__main__':
  # already an eps image
  # convert to pdf and then back to eps with cairo
  batcmd="epstopdf %s -o tmp.pdf" % (args.eps)
  result = subprocess.check_output(batcmd, shell=True)
  batcmd="pdftocairo tmp.pdf -ps tmp.ps -origpagesizes"
  result = subprocess.check_output(batcmd, shell=True)

  epsfile = open("tmp.ps","r")
  lines = epsfile.readlines()
  epsfile.close()


  temp = remove_text(lines)
  temp = remove_resources(temp)
  temp = remove_page_setup(temp)
  temp = remove_remainder(temp)

  content = ' '.join(lines)
  eps_objects = get_eps_objects(content)

  extract_graphs(eps_objects)

#  result3 = subprocess.check_output("rm -f *.eps", shell=True)

