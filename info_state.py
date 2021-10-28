import numpy as np

class InfoState4:
    def __init__(self, vector=np.zeros(shape=(4,1)), matrix=np.zeros(shape=(4,4))):
        self.vector = vector
        self.matrix = matrix

    def __add__(self, other):
        return InfoState4(self.vector + other.vector, self.matrix + other.matrix)

    def __sub__(self, other):
        return InfoState4(self.vector - other.vector, self.matrix - other.matrix)

    def clear(self):
        self.vector.fill(0)
        self.matrix.fill(0)

class InfoState3:
    def __init__(self, vector=np.zeros(shape=(3,1)), matrix=np.zeros(shape=(3,3))):
        self.vector = vector
        self.matrix = matrix

    def __add__(self, other):
        return InfoState3(self.vector + other.vector, self.matrix + other.matrix)

    def __sub__(self, other):
        return InfoState3(self.vector - other.vector, self.matrix - other.matrix)

    def clear(self):
        self.vector.fill(0)
        self.matrix.fill(0)

class InfoState6:
    def __init__(self, vector=np.zeros(shape=(6,1)), matrix=np.zeros(shape=(6,6))):
        self.vector = vector
        self.matrix = matrix

    def __add__(self, other):
        return InfoState6(self.vector + other.vector, self.matrix + other.matrix)

    def __sub__(self, other):
        return InfoState6(self.vector - other.vector, self.matrix - other.matrix)

    def clear(self):
        self.vector.fill(0)
        self.matrix.fill(0)

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    x = InfoState3()
    x.vector = rng.random((3,1))
    x.matrix = rng.random((3,3))

    y = InfoState3()
    y.vector = rng.random((3,1))
    y.matrix = rng.random((3,3))

    print("x vect", x.vector)
    print("y vect", y.vector)
    print("x+y vect", (x+y).vector)

    x += y
    print("x updated vect", x.vector)
