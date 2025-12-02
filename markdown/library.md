# Library in python
* In Python, there are many libraries you can use for free.
* To use library, you need to `import`.

## Importing library
* Simple example;
  ```python
  import numpy

  a = numpy.zeros(10)
  ```
* You can make abbriviation to library, as
  ```python
  import numpy as np

  a = np.zeros(10)
  ```

## pip
* If your computer doesn't have a library you want to use, install it using `pip`, which is a package manager.
* Any packages in PyPI (https://pypi.org/) is available.
* This shoud be done **outside** the Python interpreter or script (that is, terminal).
  ```bash
  pip install numpy
  ```
* In Jupyter notebook (Google colab), do like
  ```bash
  !pip install numpy
  ```
* To update some library,
  ```bash
  pip install numpy --update
  ```

## os module (file and directory operations)
* The `os` module provides functions for interacting with the operating system.
  ```python
  import os

  # Get current working directory
  cwd = os.getcwd()

  # Join path components
  path = os.path.join("data", "images", "photo.jpg")

  # Get file extension
  name, ext = os.path.splitext("photo.jpg")
  print(ext)  # => .jpg

  # Get absolute path
  abs_path = os.path.abspath("data")
  ```

* `os.walk` iterates through all directories and files recursively:
  ```python
  import os

  for dirpath, dirnames, filenames in os.walk("data"):
      for filename in filenames:
          full_path = os.path.join(dirpath, filename)
          print(full_path)
  ```
* This is useful for finding all files in a directory tree.
