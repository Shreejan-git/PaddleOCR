# Parent class
class ParentClass:
    def __init__(self, age):
        print("In parents: age", age)
        self.age = age + 25
        print("In parent: after adding 25", {self.age})

    def greet(self):
        return f"Hello, my father's age is {self.age}. He is 25 years older than me."


# Child class inheriting from ParentClass
class ChildClass(ParentClass):
    def __init__(self, name, age):
        # Call the constructor of the parent class
        # self.name = name
        # super().__init__(age)
        # print("after super call")
        print(age)
        self.age = age - 25
        print('Final value of age', self.age)
        super().__init__(self.age)
        print('final age', self.age)

    def introduce(self):
        return f"My name is, and I am {self.age} years old."

    def __call__(self, var):
        return f'Age + var = {self.age + var}'


# Create an instance of ChildClass
child_obj = ChildClass("Alice", 25)

# Access functionality from both ParentClass and ChildClass
print(child_obj.greet())  # Output: Hello, my name is Alice
# print(child_obj.introduce())  # Output: My name is Alice, and I am 25 years old.
# print(child_obj(25))
