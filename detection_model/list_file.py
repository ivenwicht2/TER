from os import listdir , mkdir
from os.path import isfile, join
import numpy

mypath = "data/jpeg"
onlyfiles = [f.split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f)) ]
onlyfiles = numpy.array(onlyfiles)
numpy.random.shuffle(onlyfiles)
training, test = onlyfiles[:70], onlyfiles[70:]

try : 
    mkdir('data/test')
except :
    pass
try : 
    mkdir('data/train')
except :
    pass

test_file = open("data/test/testval.txt", "w")
for ligne in test :
    write = test_file.write(str(ligne)+'\n')
test_file.close()

train_file = open("data/train/trainval.txt", "w")
for ligne in training :
    write = train_file.write(str(ligne)+'\n')
train_file.close()

