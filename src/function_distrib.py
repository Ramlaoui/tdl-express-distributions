import numpy as np
from sklearn.datasets import make_swiss_roll


class FunctionDistrib:
    def __init__(
        self, n, d, l, function_type="linear", prior="uniform", init=True, seed=42
    ):
        self.n = n
        self.d = d
        self.l = l
        self.function_type = function_type
        self.prior = prior
        self.seed = seed

        if function_type == "random":
            self.functions = self.generate_random_functions(self.l)
        else:
            assert function_type in [
                "linear",
                "rbf",
                "trigonometric",
            ], "Function type must be linear, rbf, trigonometric or random"
            self.functions = []
            input_size = d
            for i in range(self.l):
                function_layer, input_size = self.get_function(
                    function_type, input_size
                )
                self.functions.append(function_layer)

        self.generate_data()

    def get_function(self, function_type, input_size):
        if function_type == "linear":
            return self.get_linear_function(input_size)
        elif function_type == "rbf":
            return self.get_rbf_function(input_size)
        elif function_type == "trigonometric":
            return self.get_trigonometric_function(input_size)

    def get_linear_function(self, input_size):
        np.random.seed(self.seed)

        a = np.random.uniform(-1, 1, input_size)
        b = np.random.uniform(-1, 1)

        return lambda x: np.dot(a, x) + b, 1

    def get_rbf_function(self, input_size):
        np.random.seed(self.seed)

        gamma = np.random.uniform(-1, 1)
        c = np.random.uniform(-1, 1, input_size)

        return lambda x: np.exp(-gamma * np.linalg.norm(x - c) ** 2), input_size

    def get_trigonometric_function(self, input_size):
        np.random.seed(self.seed)

        a = np.random.uniform(-1, 1)

        return lambda x: np.cos(a * x) + np.sin(a * x), input_size

    def generate_random_functions(self, l):
        np.random.seed(self.seed)
        functions = []
        input_size = self.d

        for i in range(l):
            function_type = np.random.choice(["linear", "rbf", "trigonometric"])

            if function_type == "linear":
                function_layer, input_size = self.get_linear_function(input_size)
                functions.append(function_layer)
            elif function_type == "rbf":
                function_layer, input_size = self.get_rbf_function(input_size)
                functions.append(function_layer)
            elif function_type == "trigonometric":
                function_layer, input_size = self.get_trigonometric_function(input_size)
                functions.append(function_layer)

        return functions

    def apply_functions(self, Z):
        temp = Z
        for i in range(self.l):
            temp = np.apply_along_axis(self.functions[i], 1, temp).reshape(self.n, -1)

        return temp

    def generate_data(self):
        np.random.seed(self.seed)

        if self.prior == "uniform":
            Z = np.random.uniform(-1, 1, self.n * self.d).reshape(self.n, self.d)
            self.Z = Z
        elif self.prior == "gaussian":
            Z = np.random.s
            self.Z = Z
        elif self.prior == "gaussian_mixture":
            Z = np.concatenate(
                [
                    np.random.normal(-1, 0.5, self.n * self.d).reshape(self.n, self.d),
                    np.random.normal(1, 0.5, self.n * self.d).reshape(self.n, self.d),
                ]
            )
            self.Z = Z
        elif self.prior == "swiss_roll":
            Z, _ = make_swiss_roll(self.n, noise=0.1)
            self.Z = Z
        else:
            raise ValueError(
                "Prior must be uniform, gaussian, gaussian_mixture or swiss_roll"
            )

        Y = self.apply_functions(Z)
        self.Y = Y

        return Z, Y

    def resample(self):
        self.generate_data()
        return self.Z, self.Y
