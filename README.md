# hidden echoes work in progress

## Setup ZIMTOHRLI
1. clone repo
2. `cd ZIMTOHRLI && pip install .`

## setup PESQ
Might need Cython:
```bash
$ git clone https://github.com/serser/python-pesq.git
$ cd python-pesq
$ pip install . --no-build-isolation # requires numpy to build
$ cd ..
$ rm -rf python-pesq # remove the code folder since it exists in the python package folder
```

## Setup visqol

```pip install git+https://github.com/diggerdu/visqol-py.git```