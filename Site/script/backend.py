from PIL import Image
import numpy as np
import pickle 
def model(url):
    tmp = "C:\\Users\\theoo\\OneDrive\\Documents\\GitHub\\TER\Site\\image\\"
    url = url.split(r'/')[-1]
    url = tmp + url
    print("this is the url : ",url)
    im = Image.open(url)
    im = im.convert('RGB')
    im = im.resize((224,224), Image.ANTIALIAS)
    im = np.array(im)
    im = np.expand_dims(im, axis=0)
    model = load_model(r"save\model_sauvegarde")
    representation = load_model(r"save\simi_sauvegarde")
    img = np.load(r"save\img.npy")
    label = np.load(r"save\label.npy")
    Class = np.load(r"save\class.npy")
    simi = np.load(r"save\representation.npy")
    pred = model.predict(im)
    pred = pred.argmax(axis=1)[0]
    quer = representation.predict(im)[0]
    nb = 6
    distance,index = spatial.KDTree(simi).query(quer,k=nb+1)
    with open(r"save\path.txt", "rb") as fp:   # Unpickling
        path_total = pickle.load(fp)
    path = []
    for i in range(len(index)): 
        path.append(path_total[index[i]])
    return path,Class[pred]

