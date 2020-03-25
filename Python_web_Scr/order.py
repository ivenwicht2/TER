import os 
import glob
from PIL import Image
import random 

if not os.path.isdir("right") : os.mkdir("right")
if not os.path.isdir("left") : os.mkdir("left")
if not os.path.isdir("up") : os.mkdir("up")
if not os.path.isdir("down") : os.mkdir("down")

for filename in glob.glob('downloads/hieroglyphe/*.jpg'):
        im = Image.open(filename).convert("RGB")
        name = filename.split('\\')[1]

        num = random.randrange(0, 4, 1)
        if num == 0 :   
            im = im.rotate(0)
            save = "up"
        elif num == 1 : 
            im = im.rotate(90)
            save = "left"
        elif num == 2 : 
            im = im.rotate(180)
            save = "down"
        elif num == 3 : 
            im = im.rotate(260)
            save = "right"

        im.save(save+"/"+name)