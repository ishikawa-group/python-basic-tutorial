# List, tuple and dict
* *list* and *tuple* are popular types in Python to treat a set of variables.

## list
* A set of variables can be stored in *list*.
  ```python
  float_list  = [1, 2, 3]
  string_list = ["A", "B", "Three"]
  ```
* The elements in the list can be accessed with *index*.
  ```python
  float_list[0]  # => 1
  float_list[0] = 0  # replacing the element
  ```
* Note that, **in Python, the index starts with zero (not one)**; list[0] is the first element and list[1] is the second element.
* You can obtain the length of the list with `len` function.
  ```python
  a = [1, 2, 3]
  len(a)
  ```
* You can combine lists as
  ```python
  [0, 1, 2] + [3, 4]  # => [0, 1, 2, 3, 4, 5]
  ```
* Multiplication is also possible
  ```python
  a = [0] * 10  # => [0, 0, 0, ..., 0]
  ```
* You can append an element with `append`
  ```python
  a = [0, 1]
  a.append(2)
  a  # => [0, 1, 2]
  ```
* remove
  ```python
  a = [0, 1]
  a.remove(0)
  print(a)
  ```
* extend
  ```python
  a = [0, 1]
  b = [2, 3]
  c = a + b    # need to store in the different list
  a.extend(b)  # a is extended
  print(a)
  print(c)
  ```

### Index slicing
* You can access the index of the list as follows: `x[start=0 : stop=size : step=1]`
* This is called *slicing*, and in the script you do like
  ```python
  a = [1]*10
  print(a[:])    # all elements
  print(a[0:9])  # from 0 to 8 (not 9!)
  print(a[:5])   # from 0 to 4
  print(a[4:])   # from 5 to the last(10)
  ```

## Tuple
* *Tuple* is similar to a list, but different way of treating a data set.
  ```python
  a = (0, 1, 2)
  ```
* The biggest difference is that **you cannot replace the element in a tuple afterward**.
  ```python
  a = (0, 1, 2)
  a[0] = 10  # => error
  ```

## set
* `set` is a collection of unique elements (no duplicates).
  ```python
  s = {1, 2, 3}
  s = {"apple", "banana", "orange"}
  ```
* You can check if an element is in a set with `in`:
  ```python
  fruits = {"apple", "banana", "orange"}
  if "apple" in fruits:
      print("Found!")
  ```
* Sets are useful for fast membership checking:
  ```python
  valid_extensions = {".png", ".jpg", ".jpeg"}
  ext = ".png"
  if ext in valid_extensions:
      print("Valid image file")
  ```

## dict
* `dict` is a special type that allows making *key* and *value* pair.
* key and value is separated with colon (`:`).
  ```python
  d = {"Apple": 100}
  ```
* Here "Apple" is key and 100 is value. You can access the value by specifying the key.
  ```python
  d = {"Apple": 100, "Orange": 120}
  print(d["Orange"])
  ```
* You can add the key-value pair to the dict by `|` or `update` function.
  ```python
  a = {"Apple": 100}
  b = {"Banana": 200}
  c = a | b
  print(c)
  ```
  or
  ```python
  d = {"Apple": 100}
  d.update({"Banana": 200})
  print(d)
  ```
* To loop over dict,
  ```python
  d = {"Apple": 120, "Orange": 110, "Banana": 200}
  for i, j in d.items():
      print(i, "is", j, "Yen.")
  ```
* Setting initial values to dict from key can be done by `.fromkeys()`.
  ```python
  keys = {"a", "b", "c"}
  {}.fromkeys(keys, 0)  # --> {"a": 0, "b": 0, "c": 0}
  ```

## Exercise
* Create a dictionary representing a person's information (name, age, city) and access specific elements.
<a href="./answer.md#dict">answer</a>
