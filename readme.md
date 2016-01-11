
	o   o o   o o   o
	 \ /  |\  | |\  |
	  O   | \ | | \ |
	 / \  |  \| |  \|
	o   o o   o o   o

 --------------------------------
 xnn neural network library V 0.2
 --------------------------------

Authors:
Josh Susskind, Max Anger, Walter Talbott
Emotient, Inc.


## Cloning the xnn repository

```
git clone ssh://git@stash.emotient.local/res/xnn.git
```

## Building and viewing the documentation for XNN
First install sphynx into your python installation (we recommend using virtualenv).

```
pip install sphinx
pip install numpydoc
```

Then build the html docs by going to the docs directory and running the following:

```
cd docs
make html
```

You can then view the HTML documentation in a browser. For example, on OSX open it like this:

```
open docs/build/html/index.html
```


## Companion libraries

### xnn_experiments

```
git clone ssh://git@stash.emotient.local/res/xnn_experiments.git
```

This is a library for experimenting with xnn without cluttering the xnn repository itself.
It is meant as a model zoo and a sandbox for trying new ideas.
