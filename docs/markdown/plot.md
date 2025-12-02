# Making a plot in Python
* In this lecture, we will learn how to make a plot with Python. *Matplotlib*, *seaborn*, and *plotly* libraries are introduced.
* *Matplotlib* is the scientific plotting library in Python. It provides functions for making visualizations such as line charts, histograms, scatter plots, and so on.
* *Seaborn* is also widely used plotting library, which makes cooler plot.
* Plotly is gives interactive plot, which is more fancy for presentation purposes.

## matplotlib
* loading the library
  ```python
  import matplotlib.pyplot as plt
  ```

### Basic line plot
* Here is the basic usage of matplotlib.
  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(-2*np.pi, 2*np.pi, 100)
  y = np.cos(x)

  plt.plot(x,y)
  plt.xlabel("x")
  plt.ylabel("f = cos(x)")
  plt.show()
  ```
* Do not forget to call `show` function. Otherwise, plot is made but not shown.

### Scatter plot
* By changing `plot` to `scatter`, you can make a scatter plot.
  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  x = np.linspace(-2*np.pi, 2*np.pi, 100)
  y = np.cos(x)

  plt.scatter(x,y)
  plt.xlabel("x")
  plt.ylabel("f = cos(x)")
  plt.show()
  ```

### Setting limits
* The range of plotting can be set as follows.
  ```python
  plt.xlim([-2, 2])
  plt.ylim([0, 10])
  ```

### Setting labels
* x- and y-axis labels are set as follows.
  ```python
  plt.xlabel("xlabel")
  plt.ylabel("ylabel")
  ```

### Setting ticks
* Ticks can be set as follows.
  ```python
  plt.xticks(np.arange(0, 2+0.1, 9.5))
  ```

---

## Exercise
* Plot these two data in one figure for $-2\pi \le x \le 2\pi$.
1. $y = \cos(x)$
2. $y = \cos(x) + 0.5(r - 0.5)$ where $r$ is the random number of $r \in [0, 1]$.
<a href="./answer.md#plot">answer</a>

---

## Figure and Axes
* In matplotlib, there are two ways to make plots.
* One is use `plt.plot()`, which is shown above. This is in similar way to MATLAB (actually MATplotlib means MATLAB-like plot library).
* Another way to plot is use `Figure` and `Axes` objects. `Figure` controls the figure part, and `Axes` controls the axes part of each figure.
* This is more advanced, but I recommend to use it because it enables finer control of the figure.

### Example
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y)
plt.show()
```

### Setting limits
```python
ax.set_xlim([0, 1])
ax.set_xlim([0, 1])
```
### Setting labels
```python
ax.set_xlabel("x")
ax.set_ylabel("y = sin(x)")
ax.set_title("sine curve")
```

### Bar plot
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 6)
y = np.array([83, 32, 54, 22, 78])

fig, ax = plt.subplots()
ax.bar(x, y)

plt.show()
```

### Histogram
```python
import numpy as np
import matplotlib.pyplot as plt

values = 10.0 * np.random.randn(100) + 100

fig, ax = plt.subplots()
ax.hist(values, bins=10, ec="k")

plt.show()
```

### Displaying image
```python
import matplotlib.pyplot as plt

img = plt.imread("image.jpg")
fig, ax = plt.subplots()
ax.imshow(img)

plt.show()
```

### Figure configuration
```python
import numpy as np
import matplotlib.pyplot as plt

# setting font family
plt.rcParams["font.family"] = "Arial"

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

# controling the figure size and dpi
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

ax.plot(x, y)

# set x and y labels
ax.set_xlabel("x-axis", fontsize=16)
ax.set_ylabel("y = sin(x)", fontsize=16)

# when changing font size
## old ticklabels
xticklabel = ax.get_xticklabels()
yticklabel = ax.get_yticklabels()

## new ticklabels
ax.set_xticklabels(labels=xticklabel, fontsize=16)
ax.set_yticklabels(labels=yticklabel, fontsize=16)

# adjusting tick parameters
ax.tick_params(direction="in", length=10, width=1)

plt.tight_layout()
plt.show()
```

## Saving figure
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots()

ax.plot(x, y)
plt.savefig("sin.png")
plt.close()
```

# Seaborn
* `Seaborn` is another visualization library, which is based on matplotlib. This makes cooler plots than matplotlib.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

# loading dataset
iris = sns.load_dataset("iris")

# scatter plot
sns.scatterplot(data=iris)
plt.show()

# histogram
sns.histplot(iris.petal_length)
plt.show()

# pair plot -- showing all the combination
sns.pairplot(data=iris)
plt.show()
```

# Plotly
* `plotly` enables an interactive plot, which is made on the browser, and you can see numerical values when you put mouse pointer on it.
* To use plotly, you need to load the library first. Then, make `Figure` object instance and after that, you can add plots to the instance by `add_trace` method.
  ```python
  import numpy as np
  xs = np.linspace(0, 10, 100)
  sins = np.sin(xs)
  randoms = np.random.rand(100)

  import plotly.graph_objects as go
  fig = go.Figure()

  # adding scatter plot
  fig.add_trace(go.Scatter(x=xs, y=sins))

  # adding another scatter plot
  fig.add_trace(go.Scatter(x=xs, y=randoms))

  fig.show()
  ```
* If you are using Google Colab, put follwing lines after importing the library.
  ```python

  import plotly.io as pio
  pio.renderers.default = "colab"
  ```

## Application
* You can plot the interactive chart as follows.
  ```python
  import plotly
  import plotly.graph_objs as go
  from plotly.subplots import make_subplots
  import pandas as pd
  import plotly.io as pio
  pio.renderers.default = "colab"

  df = pd.read_csv(
      'https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

  fig = go.Figure(data=[
          go.Candlestick(
              x=df['Date'],
              open=df['AAPL.Open'],
              high=df['AAPL.High'],
              low=df['AAPL.Low'],
              close=df['AAPL.Close'])])
  fig.show()
  ```
