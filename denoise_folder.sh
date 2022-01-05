
wget -O model.pth https://www.dropbox.com/s/7yryykmzk06snvd/DenoiseAE_UNet_loss.pth?dl=1
python3 test_folder.py --indir $1 --outdir $2 --model model.pth