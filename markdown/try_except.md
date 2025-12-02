# try/except (error handling)
* You can handle errors using `try` and `except` statements.
  ```python
  try:
      result = 10 / 0
  except ZeroDivisionError:
      print("Cannot divide by zero!")
  ```

* You can catch multiple types of errors:
  ```python
  try:
      f = open("nonexistent.txt", "r")
  except FileNotFoundError:
      print("File not found!")
  except PermissionError:
      print("No permission to read the file!")
  ```

* Use `finally` to run code regardless of whether an error occurred:
  ```python
  try:
      f = open("test.txt", "r")
      content = f.read()
  except FileNotFoundError:
      print("File not found!")
  finally:
      print("This always runs")
  ```

* You can raise your own errors with `raise`:
  ```python
  def divide(a, b):
      if b == 0:
          raise ValueError("b cannot be zero")
      return a / b
  ```
