# If, for, and while
* There are some control statements in Python.

### if statement
* `if` statement can be written as follows
  ```python
  if a > 0:
      print("a is lager than zero")
  else:
      print("a is smaller than zero")
  ```
* You can use `else` statement to add the instructions, as
  ```python
  if a > 0:
      print("a is positive")
  elif a == 0:
      print("a is zero")
  else:
      print("a is negative")
  ```

### for statement
* In Python, the code block is expressed by *indent*.
* Either `tab` or `spaces` are OK for indent but **do not mix them**.
* `for` statement can be written as follows:
  ```python
  for i in [0, 1, 2]:
      print(i)
  ```
* `range` function is useful when using `for` statement.
  ```python
  for i in range(10):
      print(i)
  ```
* You can access the index and the value of the list with `enumerate` function.
  ```python
  a = [0, 1, 2]
  for i, val in enumerate(a):
      print(i, val)
  ```

### continue, break; the loop control
* If you want to skip the loop at some condition, you can use `continue` statement.
  ```python
  for i in range(10):
      if i == 5:
      continue
  else:
      print(i)
  ```
* The `for` loop goes to the next step when it finds `continue`. So rest of the part is not executed.
* If you want to end the loop at some condition, you can use `break` statement.
  ```python
  for i in range(10):
      if i == 5:
          break
      else:
        print(i)
  ```
* In this way, the loop after finding `i == 5` is not executed.

## Exercise
1. Write a program that checks if x is greater than y.
If x is greater, print "x is greater than y"; otherwise, print "y is greater than or equal to x".
2. Create a list containing at least five elements (numbers or strings). Use a for loop to iterate through the list and print each element.

<a href="./answer.md#control">answer</a>
