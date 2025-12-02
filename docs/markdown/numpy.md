# Numpy and scipy
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

---

# scipy
* *SciPy* is a Python library collecting scientific computing functionalities.
* It provides advanced linear algebra routines, mathematical function optimization, signal processing, special mathematical functions, and statistical distributions.

| library name         | contents                             |
| :------------------- | :----------------------------------- |
| scipy.special        | Special functions                    |
| scipy.integrate      | Integration                          |
| scipy.optimize       | Optimization                         |
| scipy.interpolate    | Interpolation                        |
| scipy.fft            | Fourier Transforms                   |
| scipy.signal         | Signal Processing                    |
| scipy.linalg         | Linear Algebra                       |
| scipy.sparse.csgraph | Sparse eigenvalue problems           |
| scipy.spatial        | Spatial data structures & algorithms |
| scipy.stats          | Statistics                           |
| scipy.ndimage        | Multidimensional image processing    |
| scipy.io             | File IO                              |

* linear algebra (linalg)
  ```python
  import numpy as np
  from scipy import linalg
  
  A = np.array([[1, 3, 2], [-1, 0, 1], [2, 3, 0]])
  
  Ainv = linalg.inv(A)

  print(Ainv)
  print(np.matmul(Ainv, A))  # matrix-matrix multiply
  ```

* numerical integration
  ```python
  import numpy as np
  from scipy import integrate

  # Define the function to integrate
  def my_func(x):
      return x**2  # Example function: x^2

  # Perform numerical integration using quad
  result, _ = integrate.quad(my_func, 0, 4)  # Integrate x^2 from 0 to 4

  print("Result of the integration:", result)
  ```

* ordinary differential equation (odeint); solving differential equation $\frac{dy}{dt} = -y$
  ```python
  import numpy as np
  from scipy.integrate import odeint
  import matplotlib.pyplot as plt

  # define function for ODE
  def func_dydt(y, t):
      dydt = -y
      return dydt

  t_list = np.linspace(0.0, 10.0, 100)
  y_init = 1.0  # initial value
  y_list = odeint(func_dydt, y_init, t_list)

  # visualization
  fig, ax = plt.subplots()
  ax.plot(t_list, y_list)
  plt.show()
  ```

---

## Exercise (scipy)
* Perform the numerical integration of $\exp(-x^2)$ function from -10 to 10, using SciPy's quad function.

<a href="./answer.md#scipy">answer</a>
