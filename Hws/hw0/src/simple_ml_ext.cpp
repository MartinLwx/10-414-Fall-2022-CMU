#include <cmath>
#include <cstdio>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void multiply(const float *X, const float *Y, float *Z, size_t m, size_t n,
              size_t p) {
  // X @ Y where X in (m, n) and Y in (n, p), store the result in Z where Z in
  // (m, p)
  for (auto i = 0; i < m; i++) {
    for (auto j = 0; j < p; j++) {
      for (auto k = 0; k < n; k++) {
        Z[i * p + j] += X[i * n + k] * Y[k * p + j];
      }
    }
  }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch) {
  /**
   * A C++ version of the softmax regression epoch code.  This should run a
   * single epoch over the data defined by X and y (and sizes m,n,k), and
   * modify theta in place.  Your function will probably want to allocate
   * (and then delete) some helper arrays to store the logits and gradients.
   *
   * Args:
   *     X (const float *): pointer to X data, of size m*n, stored in row
   *          major (C) format
   *     y (const unsigned char *): pointer to y data, of size m
   *     theta (float *): pointer to theta data, of size n*k, stored in row
   *          major (C) format
   *     m (size_t): number of examples
   *     n (size_t): input dimension
   *     k (size_t): number of classes
   *     lr (float): learning rate / SGD step size
   *     batch (int): SGD minibatch size
   *
   * Returns:
   *     (None)
   */

  auto iterations{m / batch};
  auto *logits{new float[batch * k]{}};
  auto *grad{new float[n * k]{}};

  for (auto i = 0; i < iterations; i++) {
    std::fill(logits, logits + batch * k, 0.0);
    std::fill(grad, grad + n * k, 0.0);

    multiply(&X[i * batch * n], theta, logits, batch, n, k);

    for (auto cnt = 0; cnt < batch; cnt++) {
      auto row_sum = 0.0;

      // exp
      for (auto j = 0; j < k; j++) {
        logits[cnt * k + j] = exp(logits[cnt * k + j]);
        row_sum += logits[cnt * k + j];
      }

      // compute the probs
      for (auto j = 0; j < k; j++) {
        logits[cnt * k + j] /= row_sum;
      }

      // (batch_Z - batch_y)
      logits[cnt * k + y[i * batch * 1 + cnt]] -= 1.0;
    }
    /* printf("\n"); */
    /* printf("Probs when i = %d, batch = %ld\n", i, batch); */
    /* for (auto cnt = 0; cnt < batch; cnt++) { */
    /*     for (auto j = 0; j < k; j++) { */
    /*         printf("%f ", logits[cnt * k + j]); */
    /*     } */
    /*     printf("\n"); */
    /* } */

    // get batch_x.T: (batch, n) -> (n, batch)
    auto cnt = 0;
    auto *batch_x_T{new float[n * batch]{}};
    for (auto j = 0; j < n; j++) {
      for (auto t = 0; t < batch; t++) {
        batch_x_T[cnt] = X[(i * batch + t) * n + j];
        cnt++;
      }
    }

    // logits = (batch_Z - batch_y)
    // compute batch_X.T @ (batch_Z - batch_y)
    multiply(batch_x_T, logits, grad, n, batch, k);

    // compute the gradient
    for (auto p = 0; p < n; p++) {
      for (auto q = 0; q < k; q++) {
        theta[p * k + q] -= 1.0 * lr / batch * grad[p * k + q];
      }
    }

    delete[] batch_x_T;
  }

  delete[] logits;
  delete[] grad;
  /// END YOUR CODE
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
  m.def(
      "softmax_regression_epoch_cpp",
      [](py::array_t<float, py::array::c_style> X,
         py::array_t<unsigned char, py::array::c_style> y,
         py::array_t<float, py::array::c_style> theta, float lr, int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float *>(X.request().ptr),
            static_cast<const unsigned char *>(y.request().ptr),
            static_cast<float *>(theta.request().ptr), X.request().shape[0],
            X.request().shape[1], theta.request().shape[1], lr, batch);
      },
      py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"),
      py::arg("batch"));
}
