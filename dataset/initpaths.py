# Exporting project folder to python search paths
import sys, os
folderpath  = os.path.dirname(os.path.realpath(__file__))
projectpath = os.path.split(folderpath)[0]
sys.path.insert(0, projectpath)