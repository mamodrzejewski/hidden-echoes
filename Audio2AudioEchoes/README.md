## Setup

1. Download `ArtistProtectModels/SingleEchoes` from [here](https://filedn.com/lAEQ8ShUNLzjcOukVHsWG0z/ArtistProtectModels/)

2. Clone Audio2Audioechoes required repos
```
git clone git@github.com:ctralie/ddsp.git
git clone git@github.com:ctralie/dance-diffusion.git dance_ciffusion # rename to have underscore
git clone git@github.com:ctralie/RAVE.git

cd RAVE
pip install -e .
cd ../dance_diffusion
pip install -e .
```