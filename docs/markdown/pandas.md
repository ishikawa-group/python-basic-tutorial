## Pandas

* *pandas* is a Python library for data processing and analysis. Pandas provides a great range of methods to modify and operate on data; in particular, it allows queries and joins of tables.
* In pandas, you can use various types (for example, integers, dates, floating-point numbers, and strings) in a single table.
* Another valuable tool provided by pandas is its ability to ingest from a great variety of file formats and databases, like SQL, Excel files, JSON, and comma-separated values (CSV) files.

* It is built around a data structure called the **Series** and **DataFrame**.

### loading the library
  ```python
  import pandas as pd
  ```

### Basic

#### Making a Series
* basic
  ```python
  import pandas as pd
  sales = ["David", "Bob", "Josh"]
  series = pd.Series(data=sales)
  print(series)
  ```

* You can set the column name if you like.
  ```python
  import pandas as pd
  sales  = ["David", "Bob", "Josh"]
  column = "Members"
  series = pd.Series(data=sales, name=column)
  print(series)
  ```

#### Making a DataFrame
* DataFrame is more useful than Series, so it is more often used.
* DataFrame assembles Series, so you can think like the Series is a vector and DataFrame is a matrix.
  ```python
  import pandas as pd
  sales = [["P001", "Windows", 85000],
          ["P002", "Mac", 120000],
          ["P003", "Windows", 200000],
          ["P004", "Mac", 130000]]
  columns = ["Product ID", "OS", "Price"]
  df = pd.DataFrame(data=sales, columns=columns)
  print(df)
  ```

#### Statictics
* Using simple functions
  ```python
  df["Price"].min()
  df["Price"].max()
  df["Price"].mean()
  ```

#### Sort
* ascending: from low value to high value
* descending: from high value to low value
  ```python
  df = df.sort_values(by="Price")  # ascending
  df = df.sort_values(by="Price", ascending=False)  # descending
  ```

#### Print several rows
```python
print(df.head())
```

#### Filtering: taking subgroup
```python
df_win = df[df["OS"]=="Windows"]
df_mac = df[df["OS"]=="Mac"]

df_new = df[(df["Price"] > 100000) & (df["OS"]=="Mac")]  # multiple condition
```

#### Importing from CSV (comma separated file) file
```python
import pandas as pd
df = pd.read_csv("sample.csv")
print(df.head())
```

---

## Exercise
* Suppose you have sales data in a CSV file (sales_data.csv) containing a column named "Sales".
1. Load this data into a Pandas DataFrame and see the DataFrame with `df.head()` function.
2. Calculate the total, average, max, min sales with `.sum()`, `mean()`, `max()`, and `min()` functions.
<a href="./answer.md#pandas">answer</a>

---

### Some important commands
* Let's analye the following DataFrame again.
  ```python
  import pandas as pd
  sales = [["P001", "Windows", 85000],
          ["P002", "Mac", 120000],
          ["P003", "Windows", 200000],
          ["P004", "Mac", 130000]]
  columns = ["Product ID", "OS", "Price"]
  df = pd.DataFrame(data=sales, columns=columns)
  print(df)
  ```

* Here are examples showcasing basic usages of Pandas:

#### Access specific rows using iloc (by index)
```python
specific_element = df.iloc[0, 0]
```

#### Summary statistics of numerical columns
```python
summary_stats = df.describe()
```

#### Add a new column
```python
df["2xPrice"] = df["Price"]*2
```

#### Drop a column
```python
df_new = df.drop(columns=["Price"], inplace=False)
```
* `inplace=True` will replace the df itself.

#### Taking mean value for grouped data
```python
grouped_data = df.groupby("OS").mean()

# When numeric data and categorical data are mixed
grouped_data = df.groupby("OS").mean(numeric_only=True)
```

#### Plot
```python
import matplotlib.pyplot as plt
df.plot(x="Product ID", y="Price", kind="bar")
plt.show()
```
* Following plots can be made by setting `kind` argument.
    * "line" : line plot (default)
    * "scatter": scatter plot
    * "bar" : vertical bar plot
    * "barh" : horizontal bar plot
    * "hist" : histogram
    * "box" : boxplot
    * "kde" : Kernel Density Estimation plot
    * "density" : same as ‘kde’
    * "area" : area plot
    * "pie" : pie plot

#### Exporting to CSV file
* You can export th DataFrame to a CSV file
```python
df.to_csv("computers.csv", index=False)
```

#### Reading Microsoft Excel data
* After installing xlrd by `pip install xlrd`, you can do
```python
df = pd.read_excel("water-spectra.xls")
```

---

## Exercise 2
* Let's analyze the emplyee information stored in "employee.csv", as follows.
1. Import a csv file to a Pandas DataFrame.
2. Extract the high-saraly group, whose "Salary" is higher than "55000".
3. Calculate and print the mean age of high-salary group and all the employee.
4. Calculate the mean values of each department, by using `groupby()` function.
5. Make a scatter plot of "Age" vs. "Salary".
<a href="./answer.md#pandas2">answer</a>