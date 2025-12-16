# Numpy
## numpy
* NumPy is one of the fundamental packages for scientific computing in Python.
* It contains functionality for multi-dimensional arrays, and also mathematical functions such as linear algebra operations, the Fourier transform, and pseudo-random number generators.
* Usually, numerical calculations in your code will be faster if you use numpy.

* loading library
  ```python
  import numpy as np
  ```
* `as np` is not mandatory but often used.

* Using numpy array
  ```python
  import numpy as np
  a = np.zeros(2)
  print(a)
  ```

### convert list to numpy array
* numpy array can be made by defining the list first, as
  ```python
  import numpy as np
  a = [1, 2, 3, 4, 5]
  b = np.array(a)
  print(a)
  print(b)
  ```

### Array Creation
* `numpy.zeros(n)`: array with n zeros.
* `numpy.ones(n)`: array with n ones.
* `np.arange(n)`: sequence of numbers with n elements.

#### Linear spacing
* You can have an uniformly-ditributed numbers by `linspace` function.
  ```python
  import numpy as np
  x = np.linspace(-10, 10, 100)  # start, end, number of points
  ```

### Random Number Generation
* `numpy.random.rand()`: generate random numbers from a uniform distribution.
* `numpy.random.randn()`: generate random numbers from a normal distribution.
* `numpy.random.randint()`: generate random integers.

### Mathematical functions
* Several functions are available in numpy.
    * `numpy.sin()`: sine function
    * `numpy.cos()`: cosine function
    * `numpy.exp()`: exponential
    * `numpy.log()`: natural logarithm function
    * `numpy.log10()`: base 10 logarithm function
    * `numpy.pi`: $\pi = 3.141592...$

### maximum and minimum
* Maximum and minimum values in an array can be easily found by `numpy.max` and `numpy.min` functions.
* The max/min argument i.e. the index corresponding to the max/min value is obtained by `numpy.argmax` and `numpy.argmin` functions.
  ```python
  import numpy as np
  a = [1, 2, 4, 2, 1]
  b = np.array(a)
  print(np.max(b))
  print(np.argmax(b))
  ```

---

## Exercise (numpy)
* Let's say you have sales data for a week represented as a NumPy array. Calculate the total sales for the week.
<a href="./answer.md#numpy">answer</a>
