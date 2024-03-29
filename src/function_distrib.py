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
        function_type=None,
        random=True,
        output_function=None,
        prior="uniform",
        init=True,
        function_seed=42,
        seed=42,
        is_debug=False,
    ):
        self.n = n
        self.d = d
        self.l = l
        self.output_size = output_size
        self.function_type = function_type
        self.output_function = output_function
        self.prior = prior
        self.seed = seed
        self.function_seed = function_seed
        self.i = 0
        self.is_debug = is_debug

        if self.function_type == "random":
            self.function_type = [
                "linear",
                "quadratic",
                "rbf",
                "trigonometric",
                "sigmoid",
                "relu",
            ]
        elif "-" in self.function_type:
            self.function_type = self.function_type.split("-")
            if not (random):
                self.l = len(self.function_type)

        if self.output_function is not None and type(self.function_type) != list:
            print("Output function can only be used with random function type")

        if type(self.function_type) == list and random:
            self.functions, self.function_details = self.generate_random_functions(
                self.l, output_function=self.output_function
            )
        else:
            self.functions = []
            self.function_details = []
            input_size = d
            for i in range(self.l):
                if type(self.function_type) == list:
                    function_type = self.function_type[i]
                if i == self.l - 1:
                    output_size = self.output_size
                else:
                    output_size = None
                function_layer, input_size, function_detail = self.get_function(
                    function_type, input_size, output_size
                )
                self.functions.append(function_layer)
                self.function_details.append(function_detail)
                self.i += 1

        self.generate_data()

    def get_function(self, function_type, input_size, output_size):
        if function_type == "linear":
            return self.get_linear_function(input_size, output_size)
        elif function_type == "quadratic":
            return self.get_quadratic_function(input_size, output_size)
        elif function_type == "sigmoid":
            return self.get_sigmoid_function(input_size, output_size)
        elif function_type == "relu":
            return self.get_relu_function(input_size, output_size)
        elif function_type == "rbf":
            return self.get_rbf_function(input_size, output_size)
        elif function_type == "trigonometric":
            return self.get_trigonometric_function(input_size, output_size)

    def get_linear_function(self, input_size, output_size=None):
        np.random.seed(self.function_seed + self.i)

        if output_size is None:
            output_size = np.random.randint(1, 4)
        A = np.random.uniform(-10, 10, (output_size, input_size))
        b = np.random.uniform(-10, 10, output_size)

        return lambda x: A.dot(x) + b, output_size, {"type": "linear", "A": A, "b": b}

    def get_quadratic_function(self, input_size, output_size=None):
        np.random.seed(self.function_seed + self.i)

        if output_size is not None:
            print("Warning: output_size is not used for quadratic functions")

        A = np.random.uniform(-10, 10, (input_size, input_size))
        b = np.random.uniform(-10, 10, (1, input_size))
        c = np.random.uniform(-10, 10)

        return (
            lambda x: x.T.dot(A).dot(x) + b.dot(x) + c,
            1,
            {"type": "quadratic", "A": A, "b": b, "c": c},
        )

    def get_sigmoid_function(self, input_size, output_size=None):
        np.random.seed(self.function_seed + self.i)

        if output_size is not None:
            print("Warning: output_size is not used for sigmoid functions")

        return lambda x: 1 / (1 + np.exp(-x)), input_size, {"type": "sigmoid"}

    def get_relu_function(self, input_size, output_size=None):
        np.random.seed(self.function_seed + self.i)

        if output_size is not None:
            print("Warning: output_size is not used for relu functions")

        return lambda x: np.maximum(x, 0), input_size, {"type": "relu"}

    def get_rbf_function(self, input_size, output_size):
        np.random.seed(self.function_seed + self.i)

        if output_size is not None:
            print("Warning: output_size is not used for rbf functions")

        gamma = np.random.uniform(0, 5)
        c = np.random.uniform(-1, 1, input_size)

        return (
            lambda x: np.exp(-gamma * np.linalg.norm(x - c) ** 2),
            1,
            {
                "type": "rbf",
                "gamma": gamma,
                "c": c,
            },
        )

    def get_trigonometric_function(self, input_size, output_size):
        np.random.seed(self.function_seed + self.i)

        if output_size is not None:
            print("Warning: output_size is not used for trigonometric functions")

        a = np.random.uniform(-1, 1)

        return (
            lambda x: np.cos(a * x) + np.sin(a * x),
            input_size,
            {
                "type": "trigonometric",
                "a": a,
            },
        )

    def generate_random_functions(self, l, output_function=None):
        np.random.seed(self.function_seed)
        functions = []
        function_details = []
        previous_function = None
        function_type_rotation = self.function_type
        input_size = self.d

        for i in range(l):
            if i == l - 1:
                output_size = self.output_size
            else:
                output_size = None

            if output_function == "sigmoid" and i == l - 2:
                function_type = "linear"
                output_size = self.output_size
            elif output_function is not None and i == l - 1:
                function_type = output_function
            else:
                if previous_function is None:
                    sample_from = self.function_type
                else:
                    sample_from = [
                        function
                        for function in function_type_rotation
                        if function != previous_function
                    ]
                function_type = np.random.choice(sample_from)
                if function_type == "rbf":
                    function_type_rotation = [
                        f for f in self.function_type if f != "rbf"
                    ]
                previous_function = function_type

            if self.is_debug:
                print(f"function_type: {function_type}")
                print(f"input_size: {input_size}")

            function_layer, input_size, function_detail = self.get_function(
                function_type, input_size, output_size
            )
            functions.append(function_layer)
            function_details.append(function_detail)

            if self.is_debug:
                print(f"output_size: {output_size}")

        return functions, function_details

    def apply_functions(self, Z, n=None, visualize=False):
        temp = Z
        if n is None:
            n = self.n
        for i in range(self.l):
            if visualize:
                print(f"Comp. layer {i} - temp.shape: {temp.shape}")
                if temp.shape[1] <= 2:
                    self.plot(temp)
                    plt.show()
            temp = np.apply_along_axis(self.functions[i], 1, temp).reshape(n, -1)
        if visualize:
            print(f"Comp. layer {i} - temp.shape: {temp.shape}")
            self.plot(temp)
            plt.show()

        return temp

    def generate_data(self, n=None, visualize=False):
        np.random.seed(self.seed + self.i)
        self.i += 1
        if n is None:
            n = self.n

        if self.prior == "uniform_cube":
            Z = np.random.uniform(-1, 1, n * self.d).reshape(n, self.d)
            self.Z = Z
        elif self.prior == "uniform_ball":
            random_directions = np.random.normal(size=(n, self.d))
            norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
            unit_vectors = random_directions / norms
            random_radii = np.random.rand(n) ** (1.0 / self.d)
            Z = unit_vectors * random_radii[:, np.newaxis]
            self.Z = Z
        elif self.prior == "gaussian":
            Z = np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), n)
            self.Z = Z
        elif self.prior == "gaussian_mixture":
            Z, _ = make_blobs(n_samples=n, n_features=self.d, centers=2)
            self.Z = Z
        else:
            raise ValueError(
                "Prior must be uniform, gaussian, gaussian_mixture or swiss_roll"
            )

        Y = self.apply_functions(self.Z, n=n, visualize=visualize)
        self.Y = Y

        return Z, Y

    def get_function_types(self):
        for i, function_detail in enumerate(self.function_details):
            print(f"Layer {i}:")
            for k, v in function_detail.items():
                print(f"  {k}: {v}")

    def plot(self, to_plot=None, plot="input", ax=None, n_bins=20):
        if to_plot is None:
            if plot == "input":
                to_plot = self.Z
            elif plot == "output":
                to_plot = self.Y
        if to_plot.shape[1] == 1:
            return self.plot_1d(to_plot, plot, ax, n_bins)
        elif to_plot.shape[1] == 2:
            return self.plot_2d(to_plot, plot, ax, n_bins)
        else:
            raise ValueError("Can only plot 1D or 2D data")

    def plot_1d(self, to_plot=None, plot="input", ax=None, n_bins=20):
        if to_plot is None:
            if plot == "input":
                to_plot = self.Z
            elif plot == "output":
                to_plot = self.Y
        if to_plot.shape[1] != 1:
            raise ValueError("Can only plot 1D data")
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hist(to_plot, bins=n_bins, density=True)
        return ax

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

        # Compute bin centers
        xpos, ypos = np.meshgrid(
            (xedges[:-1] + xedges[1:]) / 2,
            (yedges[:-1] + yedges[1:]) / 2,
            indexing="ij",
        )

        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Compute the width and depth of the bars
        dx = np.diff(xedges) / 2
        dy = np.diff(yedges) / 2
        dx = np.repeat(dx, n_bins)
        dy = np.repeat(dy, n_bins)

        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average")
        return ax

    def plot_input_output(
        self, X_input=None, X_output=None, ax=None, n_bins=20, title=True
    ):
        if X_input is None:
            X_input = self.Z
        if X_output is None:
            X_output = self.Y
        if ax is None:
            fig, ax = plt.subplots(
                1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}
            )
            if X_input.shape[1] == 1:
                ax[0].remove()
                ax[0] = fig.add_subplot(121)
            if X_output.shape[1] == 1:
                ax[1].remove()
                ax[1] = fig.add_subplot(122)
        ax[0] = self.plot(to_plot=X_input, plot="input", ax=ax[0], n_bins=n_bins)
        ax[1] = self.plot(to_plot=X_output, plot="output", ax=ax[1], n_bins=n_bins)
        if title:
            ax[0].set_title("Input distribution")
            ax[1].set_title("Output distribution")
        return ax

    def resample(self, n=None):
        if n is None:
            n = self.n
        self.generate_data(n)
        return self.Z, self.Y
