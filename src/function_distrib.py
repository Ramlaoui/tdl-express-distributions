import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class FunctionDistrib:
    def __init__(
        self,
        n,
        d,
        l,
        output_size=None,
        function_type="linear",
        prior="uniform",
        init=True,
        seed=42,
    ):
        self.n = n
        self.d = d
        self.l = l
        self.output_size = output_size
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
                if i == self.l - 1:
                    output_size = self.output_size
                else:
                    output_size = None
                function_layer, input_size = self.get_function(
                    function_type, input_size, output_size
                )
                self.functions.append(function_layer)

        self.generate_data()

    def get_function(self, function_type, input_size, output_size):
        if function_type == "linear":
            return self.get_linear_function(input_size, output_size)
        elif function_type == "rbf":
            return self.get_rbf_function(input_size, output_size)
        elif function_type == "trigonometric":
            return self.get_trigonometric_function(input_size, output_size)

    def get_linear_function(self, input_size, output_size=None):
        np.random.seed(self.seed)

        if output_size is None:
            output_size = np.random.randint(1, 4)
        A = np.random.uniform(-1, 1, (output_size, input_size))
        b = np.random.uniform(-1, 1, output_size)

        return lambda x: A.dot(x) + b, output_size

    def get_rbf_function(self, input_size, output_size):
        np.random.seed(self.seed)

        if output_size is not None:
            print("Warning: output_size is not used for rbf functions")

        gamma = np.random.uniform(-1, 1)
        c = np.random.uniform(-1, 1, input_size)

        return lambda x: np.exp(-gamma * np.linalg.norm(x - c) ** 2), input_size

    def get_trigonometric_function(self, input_size, output_size):
        np.random.seed(self.seed)

        if output_size is not None:
            print("Warning: output_size is not used for trigonometric functions")

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

        if self.prior == "uniform_cube":
            Z = np.random.uniform(-1, 1, self.n * self.d).reshape(self.n, self.d)
            self.Z = Z
        elif self.prior == "uniform_ball":
            random_directions = np.random.normal(size=(self.n, self.d))
            norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
            unit_vectors = random_directions / norms
            random_radii = np.random.rand(self.n) ** (1.0 / self.d)
            Z = unit_vectors * random_radii[:, np.newaxis]
            self.Z = Z
        elif self.prior == "gaussian":
            Z = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), self.n)
            self.Z = Z
        elif self.prior == "gaussian_mixture":
            Z, _ = make_blobs(n_samples=self.n, n_features=self.d, centers=2)
            self.Z = Z
        else:
            raise ValueError(
                "Prior must be uniform, gaussian, gaussian_mixture or swiss_roll"
            )

        Y = self.apply_functions(Z)
        self.Y = Y

        return Z, Y

    def plot_2d(self, to_plot=None, plot="input", ax=None, n_bins=20):
        if to_plot is None:
            if plot == "input":
                to_plot = self.Z
            elif plot == "output":
                to_plot = self.Y
        if to_plot.shape[1] != 2:
            raise ValueError("Can only plot 2D data")
        # Plot the histograms of the 2D data (3d plot)
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        hist, xedges, yedges = np.histogram2d(
            to_plot[:, 0], to_plot[:, 1], bins=n_bins, density=True
        )
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average")
        return ax

    def plot_input_output(
        self, X_input=None, X_output=None, ax=None, n_bins=20, title=True
    ):
        if ax is None:
            fig, ax = plt.subplots(
                1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}
            )
        ax[0] = self.plot_2d(to_plot=X_input, plot="input", ax=ax[0], n_bins=n_bins)
        ax[1] = self.plot_2d(to_plot=X_output, plot="output", ax=ax[1], n_bins=n_bins)
        if title:
            ax[0].set_title("Input distribution")
            ax[1].set_title("Output distribution")
        return ax

    def resample(self):
        self.generate_data()
        return self.Z, self.Y
