from google_images_download import google_images_download
import os, os.path

#instantiate the class
loop = 0
img = 0
while img < 100 :
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords":"hieroglyphe","limit":100,"print_urls":True,"extensions" : "jpg",}
    paths = response.download(arguments)
    #print complete paths to the downloaded images
    print(paths)
    try :
        img = len(os.listdir('downloads/hieroglyphe'))
    except :
        img = 0
    loop += 1


print("number of loop = ",loop)