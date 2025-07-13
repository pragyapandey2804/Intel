# -*- coding: utf-8 -*-


from PIL import Image, ImageFile,ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
import glob, os



X=[]

os.chdir(os.path.abspath("x_train/1"))
for file in glob.glob("*.png"):
    image = Image.open(file)
    width, height = image.size
    if [width,height] not in X:
        X.append([width,height]) 
os.chdir(os.path.abspath("../2"))

for file in glob.glob("*.png"):
    image = Image.open(file)
    width, height = image.size
    if [width,height] not in X:
        X.append([width,height]) 
        
    
print(X)

min=3000
j=0
for k in range(len(X)):
    w,h=X[k]
    if w<min:
        min=w
        j=k

taille=X[j]
print(taille) 
taille=[x//6 for x in taille]
    



os.chdir(os.path.abspath("../.."))
List_dir=["Training_images", "Test_images"]
for path in List_dir:    
    if not os.path.exists(path):
        os.mkdir(path)


regex1 = re.compile("(.*)_(.*)_00_1_._(.*)")
regex2 = re.compile("(.*)_(.*)_00_2_._(.*)")
regex3 = re.compile("(.*)_(.*)_00_3_._(.*)")
regex4 = re.compile("(.*)_(.*)_00_4_._(.*)")
os.chdir(os.path.abspath("x_train/1"))

for file in glob.glob("*.png"):
    image = Image.open(file)
    image = image.resize(taille)
    

    if(bool(re.match(regex1, file))):
        image = image.rotate(170)
        image=ImageOps.mirror(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex2, file))):
        image = image.rotate(-89) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex3, file))):
        image=ImageOps.flip(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    elif(bool(re.match(regex4, file))):
        image = image.rotate(-82) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    image.save(os.path.abspath("../..")+"/"+List_dir[0]+"/"+file,'JPEG')


os.chdir(os.path.abspath("../2"))

for file in glob.glob("*.png"):
    image = Image.open(file)
    image = image.resize(taille)
    
    if(bool(re.match(regex1, file))):
        image = image.rotate(170)
        image=ImageOps.mirror(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex2, file))):
        image = image.rotate(-89) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex3, file))):
        image=ImageOps.flip(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    elif(bool(re.match(regex4, file))):
        image = image.rotate(-82) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    image.save(os.path.abspath("../..")+"/"+List_dir[0]+"/"+file,'JPEG')
    
    

os.chdir(os.path.abspath("../../x_test"))

for file in glob.glob("*.png"):
    image = Image.open(file)
    image = image.resize(taille)
    
    if(bool(re.match(regex1, file))):
        image = image.rotate(170)
        image=ImageOps.mirror(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex2, file))):
        image = image.rotate(-89) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 
    elif(bool(re.match(regex3, file))):
        image=ImageOps.flip(image)
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    elif(bool(re.match(regex4, file))):
        image = image.rotate(-82) 
        width, height=image.size
        ratio_lar=0.06
        ratio_lon=0.45
        left=width/2-width*ratio_lar
        right=width/2+width*ratio_lar
        top=height/2-height*ratio_lon
        bottom=height/2+height*ratio_lon
        image = image.crop((left, top, right, bottom)) 

    image.save(os.path.abspath("..")+"/"+List_dir[1]+"/"+file,'JPEG')
