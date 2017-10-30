# Problem Statement
Using .ps files scanned from research paper archives, we want to run the images through a Convolutional Neural Network to label images as graph or non-graph.

## Instructions
cd to the code/ directory and run "python create_data_and_run_cnn.py [directory of .ps files]".

### Notes
- name of files that are graphs will be in code/results/graphs_output.txt, and non-graphs in code/results/nongraphs_output.txt
- convert_ps2jpg.py will output the .jpg files in the same directory as input .ps files (these are not deleted upon cleanup)
