import torch 
from PIL import Image
import os
import numpy as np 
import torchvision.transforms.functional as TF
from scipy import spatial
import pickle

def extraction(model,image):

    output = model.forward(image)
    return output.cpu().detach().numpy()

if __name__ == "__main__":
    model = torch.load("save/model")
    phys = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(phys)
    model.eval()

    all_simi = []
    all_path = []

    folder = "../stock_image"
    for dirname, dirs, files in os.walk(folder):
        for filename in files:
            filename_without_extension, extension = os.path.splitext(filename)
            if extension == '.jpg':
                dirname = dirname.split('\\')[-1]
                dirname = dirname.split('/')[-1]
                path = folder+'/'+ dirname+"/"+filename
                img = Image.open(path)
                img = TF.to_tensor(img).to(phys)
                img.unsqueeze_(0)
                u_simi = extraction(model,img)
                all_simi.append(u_simi[0])
                all_path.append(dirname+'/'+filename)

    all_simi = np.array(all_simi)
    all_simi = spatial.KDTree(all_simi)
    print(type(all_simi))

    with open('save/simi', 'wb') as handle:
        pickle.dump(all_simi, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('save/path_simi', 'wb') as handle:
        pickle.dump(all_path, handle, protocol=pickle.HIGHEST_PROTOCOL)

 
