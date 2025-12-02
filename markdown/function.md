# Function
* In this lecture, we learn how to define and use functions in Python.
* A **function** summarizes several instructions (or commands) into one.
* First you **define** a function, then you **call** it to use.
* Let's try to calculate the average score of Math, English, and Science for students A and B.
  ```python
  stu_A_score = {"Math": 89, "English": 30, "Science": 86}
  stu_B_score = {"Math": 59, "English": 70, "Science": 65}

  aveA = (stu_A_score["Math"] + stu_A_score["English"] + stu_A_score["Science"])/3
  aveB = (stu_B_score["Math"] + stu_B_score["English"] + stu_B_score["Science"])/3

  print(f"Average of A is {aveA:.2f}, B is {aveB:.2f}")
  ```

* Clearly, taking average is common procedure so you do not want to write it twice.
  ```python
  def average(scores):
      sum = 0
      for key, val in scores.items():
          sum += val
      ave = sum/len(scores)
      return ave

  stu_A_score = {"Math": 89, "English": 30, "Science": 86}
  stu_B_score = {"Math": 59, "English": 70, "Science": 65}

  aveA = average(stu_A_score)
  aveB = average(stu_B_score)

  print(f"Average of A is {aveA:.2f}, B is {aveB:.2f}")
  ```

## Basic
* You can define a function like
  ```python
  def say_hello():
      return "hello"
  ```

* A function will `return` the result when it is called, and returned value will be stored to other variable.
  ```python
  def say_hello():
      return "hello"

  s = say_hello()
  print(s)
  ```

* The `return` statement is not mandatory, so you can omit it like
  ```python
  def say_hello2():
      print("hello")

  say_hello2()
  ```

* But I recommend writing return for safety. If no need to return, you can put `None`.
  ```python
  def say_hello3():
      print("hello")
      return None

  say_hello3()
  ```

## Exercise
* Define and call the function that repeats the string you inputted ("abc" -> "abcabc")

<a href="./answer.md#function1">answer</a>

## Argument
* **argument** is the variable that one passes to the function. The function usually does some procedure using that argument. You can pass an argument to the function, like
  ```python
  def say_something(s):
      print(s)
      return None

  say_something("hello")
  ```

* You can set the default value for arguments like
  ```python
  def say_something(s="hello"):
      print(s)
      return s

  say_something("good morning")
  say_something()
  ```

* You can use multiple arguments
  ```python
  def combine_string(s1, s2):
      new_string = s1 + " " + s2
      return new_string

  s = combine_string("Good", "Morning")
  print(s)
  ```

* You can also use multiple return-value, as
  ```python
  def say_strong(s1, s2):
      s1new = s1 + "!"
      s2new = s2 + "!"
      return s1new, s2new

  s1new, s2new = say_strong("Good", "Morning")
  print(s1new, s2new)
  ```

## Scope of variables
* When you define the function, you have to know about the *scope* of variables.
* The variables defined *inside* the function cannot be used outside the function.
  ```python
  def hello():
    i = 10
    print(i)
    return

  hello()   # => 10
  print(i)  # => not defined
  ```

* This is because `i` in the above funcion is *local variable*.
* Instead, variables defined outside the function is *global variable* so they can be accessed even inside the function.
  ```python
  i = 10
  def hello():
      print(i)
      return

  hello()
  ```

---

## Exercise
1. Create a function that takes two numbers as arguments and returns their sum.
2. Create a function that recieves a list (e.g. [10, 20, 30]), and then returns the new list in which each elements are doubled.
<a href="./answer.md#function2">answer</a>
