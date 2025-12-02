# Class in python

## Object-oriented programming in Python
In non-object-oriented programming, the function to treat some data is independent from the data structure. For example, the multiplication of scalar, vector, and matrix is all different procedures.
* Therefore you have to write functions for each; `multiply_scaler`, `multiply_vector`, and `multiply_matrix`, for example.
* In object-oriented programming, the function to treat the data (object) is defined with the data itself; *object is first, and its function should belong to it*.
Usually, objected-oriented programming has some merits, for example, it is easier to read, and more robust to error (wrong coding).

## Class
* Class is a template for an object. You need to define *data structure* and functions. 
* Data structure or variables belonging to a class is called *attribute*. Functions are usually called *methods*.
* If you define a class and make some object in the main routine, it is called *instantiation*.

---

## Defining a class in Python
* Basics:
  ```python
  class SimpleData:
      a = 0
      b = 0

      def sum(self):
          c = self.a + self.b
          return c
  
      def set(self, a, b):
          self.a = a
          self.b = b
  ```
* Class needs *at least one argument*, so `self` is used for this purpose. It specifies the instance itself, and *should be always included*.
* `sum` and `set` are methods belonging to `SimpleData`.
* In `set`, *instance variables* a and b are set.

* You can call the above class in the main routine, as
  ```python
  data1 = SimpleData()
  data1.set(1, 2)
  print(data1.sum())
  ```
* You can access the variables belonging to the object by `.` (dot). This is called an *instance variable*.
  ```python
  a1 = SimpleData.a
  b1 = SimpleData.b
  ```

### special methods
1. `__init__`
* This is called *constructor*, which is the special method called when the class is instantiated. It should be written like
  ```python
  def __init__(self):
      self.a = 0
      self.b = 0
  ```

2. `__call__`
* This is called when the instance is called like a function.
  ```python
  class Hello:
      def __init__(self):
          print("init is called")

      def __call__(self):
          print("call is called")

  hello = Hello()
  hello()
  ```

3. `__len__`
* This is called when `len()` is used on the instance.
  ```python
  class MyList:
      def __init__(self, items):
          self.items = items

      def __len__(self):
          return len(self.items)

  my_list = MyList([1, 2, 3, 4, 5])
  print(len(my_list))  # => 5
  ```

4. `__getitem__`
* This is called when accessing elements with `[]` (indexing).
  ```python
  class MyList:
      def __init__(self, items):
          self.items = items

      def __len__(self):
          return len(self.items)

      def __getitem__(self, idx):
          return self.items[idx]

  my_list = MyList([10, 20, 30])
  print(my_list[0])  # => 10
  print(my_list[2])  # => 30
  ```
* These methods are commonly used in PyTorch Dataset classes.

---

## Exercise
* Create a Car class with the following specifications:

1. Attributes:
* maker: Maker of the car (string)
* model: Model of the car (string)
* year: Year the car was manufactured (integer)

2. Methods:
* `__init__`: Constructor method to initialize the attributes.
* display_info: Method that prints out the make, model, and year of the car.
* Create an instance of the Car class, set some values for its attributes, and display its information.

<a href="./answer.md#class">answer</a>

---

## super
* Now consider making a class for a soccer player and a baseball player.
  ```python
  class SoccerPlayer:
      def __init__(self, name, age):
          self.name = name
          self.age = age

      def self_introduce(self):
          print(f"I'm {self.name}, {self.age} years old.")

      def kick(self):
          print("Kick!")

  class BaseballPlayer:
      def __init__(self, name, age):
          self.name = name
          self.age = age

      def self_introduce(self):
          print(f"I'm {self.name}, {self.age} years old.")

      def catch(self):
          print("Catch!")
  ```
* The `self_introduce` method is common to two classes, so we would like to omit to write it twice.
* To do this, let's define `Person` class, and consider `SoccerPlayer` and `BaseballPlayer` inherit the nature of `Person` class. This is called **inheritance**.
  ```python
  class Person:
      def __init__(self, name, age):
          self.name = name
          self.age = age

      def self_introduce(self):
          print(f"I'm {self.name}, {self.age} years old.")

  # put Person as the argument of the class definition
  class SoccerPlayer(Person):
      def kick(self):
          print("Kick!")

  class BaseballPlayer(Person):
      def catch(self):
          print("Catch!")

  s1 = SoccerPlayer("Messi", 36)
  b1 = BaseballPlayer("Okada", 66)

  # Now SoccerPlayer and BaseballPlayer classes can use 
  # self_introduce, because they inherit the Person class.
  s1.self_introduce()
  b1.self_introduce()
  ```
* You can add some attributes specialized to the child class by calling `super().__init__()` in `__init__`.
  ```python
  class Person:
      def __init__(self, name, age):
          self.name = name
          self.age = age

      def self_introduce(self):
          print(f"I'm {self.name}, {self.age} years old.")

  class SoccerPlayer(Person):
      def __init__(self, name, age, is_striker):
          # call __init__ of the base class
          super().__init__(name, age)
          # is_striker attribute is only for SoccerPlayer
          self.is_striker = is_striker

      def self_introduce(self):
          super().self_introduce()
          print("I'm a soccer player.")

          if self.is_striker:
              print("I'm a striker.")

      def kick(self):
          print("Kick!")

  s1 = SoccerPlayer("Messi", 36, True)

  s1.self_introduce()
  ```
