#!/bin/sh
export PYTHONPATH="$PYTHONPATH:/Users/admin/Documents/gmsh-git-MacOSX-SDK/lib"
python ccore.py -clscale 21
#for i in {1..50}
#do
#    python ccore.py -clscale $i
#done
