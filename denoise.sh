
wget -O model.pth https://www.dropbox.com/s/4j7os8krnjmynbn/Deblur_UNet.pth?dl=1
python3 test.py --inimage $1 --outimage $2 --model model.pth
