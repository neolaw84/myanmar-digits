# Myanmar Digits

[MNIST](http://yann.lecun.com/exdb/mnist/)-like dataset containing Myanmar (Burmese) hand-written digits. 

## Installation

```bash
git clone git@github.com:neolaw84/myanmar-digits.git
cd myanmar-digits
pip install -r requirements.txt
```

> `pip install myanmar-digits` is To-Do

> This requires python version >= 3.7.

## Usage

```python
# make sure myanmar_digits is visible in path

from myanmar_digits import load_data

X, y = load_data(return_X_y=True)
```

## `raw_data` to `labelled_data`

### Chopping up

Raw data is given in `raw_data` directory. The format of the images here is to have 10 columns, preferrably evenly spaced, with the first column containing Burmese 0 (၀ or zero) and the last column containing Burmese 9 (၉ or nine). 

The following command will create `chopped_data` directory with images chopped up to individual digits and `chopped_data_info.csv`. 

```bash
python myanmar_digits/utils.py chop_images 
```

> Make sure `chopped_data` directory is empty.

> It will overwrite `chopped_data_info.csv` if it exists.

### Pseudo labelling

Based on the position of the written digit in each image, there is a rudimentary labelling facility to put the `chopped_data` into `pseudo_label_data/<label>/<images>`. 

Use the following command

```bash
python myanmar_digits/utils.py prepare_pseudo_label_data
```

### Manual labelling

Afterwards, try to clean-up (delete) and/or move the images around in `pseudo_label_data` and change its name to `labelled_data`.

### Pickling

Finally, you can pickle all data using the following command:

```bash
python myanmar_digits/utils.py pickle_data --pickle-protocol 3
# pickle-protocol is given as 3 to have python 3.7 compatible
```