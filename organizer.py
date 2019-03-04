from os import listdir
from os.path import isfile, join
import shutil,sys
import os

mypath = 'test_sc5'
allfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

with open('ground_truth.txt', 'r') as f:
    raw_data = f.read()
    f.close()
raw_data = raw_data.strip()
lista_data = raw_data.split('\n')

for e in lista_data:
    blocchi = e.split(';')
    blocchi[1] = blocchi[1].replace(" ", "")
    blocchi[1] = blocchi[1].replace("/", "")
    blocchi[0] = blocchi[0].replace(" ", "")
    if not os.path.exists('test_folder/'+blocchi[1]):
        os.makedirs('test_folder/'+blocchi[1])
    #aggiungi foto alla cartella
    shutil.copy('test_sc5/'+ blocchi[0], 'test_folder/'+ blocchi[1])
