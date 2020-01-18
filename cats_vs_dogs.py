import os, shutil

SRC = r"D:\Working\CATS_VS_DOGS\cats_and_dogs_labelled"
DIR = r"D:\Working\CATS_VS_DOGS\cats_vs_dogs_small"

#create directories
for dir in ("train", "validation", "test"):
    for subdir in ("cats", "dogs"):
        dirname = os.path.join(DIR, dir, subdir)
        os.makedirs(dirname)  # exist_ok=True
        
numberings = (0, 1000, 1500, 2000)
numberings = (0, 10, 15, 20)   # for test purposes

for k in ("cat", "dog"):
    for (j,setname) in enumerate(["train", "validation", "test"]):
        r = range(*numberings[j:j+2])
        g = ("cat.{}.jpg".format(i) for i in r)  #filenames
        #copy each file
        for filename in g:
            src = os.path.join(SRC, filename)
            dst = os.path.join(DIR, setname, k+'s', filename)
            shutil.copyfile(src, dst)
        #after copying files into a given folder: print how many files are in that folder
        n = len(os.listdir(os.path.join(DIR, setname, k+'s')))
        print("{}/{}:".format(setname, k+'s').ljust(17) + "{: >4} images".format(n))
else: print("FINISHED COPYING IMAGES")
            
