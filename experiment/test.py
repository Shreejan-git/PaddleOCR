class ParentClass:
    def __init__(self, age):
        self.age = age + 25
        print("In parent: age is", {self.age})


class ChildClass(ParentClass):
    def __init__(self, age):
        self.age = age + 100
        super().__init__(self.age)
        super().__init__(age)
        # self.age = 40
        print(f"The value of age using self is:{age}")
        print(f"The value of age without using self is: {age}")

    def __call__(self, var):
        return f'Age + var = {self.age + var}'


if __name__ == "__main__":
    child_obj = ChildClass(25)

    print(child_obj)
