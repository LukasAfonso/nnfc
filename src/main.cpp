#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>

#define MAT_AT(mat, i, j) (mat).values[assert(i < (mat).rows && j < (mat).cols && i >= 0 && j >= 0), (i) * (mat).cols + (j)]
#define MAT_FREE(mat) if ((mat).values) { free((mat).values); (mat).values = nullptr; }

typedef struct
{
    int rows, cols;
    double *values;
} Mat;

typedef struct
{
    size_t count;
    Mat *items;
} Weights;

typedef struct
{
    size_t count;
    Mat *items;
} Traces;

inline Mat allocate_mat(int rows, int cols)
{
    Mat mat = {rows, cols, nullptr};
    mat.values = static_cast<double*>(malloc(sizeof(*mat.values) * rows * cols));
    return mat;
}

inline void fill_mat(const Mat &mat, double value)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(mat, i, j) = value;
        }
    }
}

inline std::string str_mat(Mat mat)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            ss << MAT_AT(mat, i, j) << " ";
        }
        ss << std::endl;
    }
    return ss.str();
}

inline void print_mat(const Mat mat)
{
    std::cout << str_mat(mat) << std::endl;
}

inline Mat dot(const Mat& mat1, const Mat& mat2)
{
    assert(mat1.cols == mat2.rows);
    const Mat res = allocate_mat(mat1.rows, mat2.cols);

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat2.cols; ++j)
        {
            double sum = 0;
            for (int k = 0; k < mat1.cols; ++k)
            {
                sum += MAT_AT(mat1, i, k) * MAT_AT(mat2, k, j);
            }
            MAT_AT(res, i, j) = sum;
        }
    }

    return res;
}

inline Mat add(const Mat& mat1, const Mat& mat2)
{
    if (mat1.rows != mat2.rows || mat1.cols != mat2.cols)
    {
        std::fprintf(stderr, "Mat1: %dx%d / Mat2: %dx%d\n", mat1.rows, mat1.cols, mat2.rows, mat2.cols);
        __asm("int3");
    }
    const Mat res = allocate_mat(mat1.rows, mat1.cols);

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            MAT_AT(res, i, j) = MAT_AT(mat1, i, j) + MAT_AT(mat2, i, j);
        }
    }

    return res;
}

inline Mat mult(const Mat& mat1, const Mat& mat2)
{
    assert(mat1.rows == mat2.rows);
    assert(mat1.cols == mat2.cols);
    const Mat res = allocate_mat(mat1.rows, mat1.cols);

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            MAT_AT(res, i, j) = MAT_AT(mat1, i, j) * MAT_AT(mat2, i, j);
        }
    }

    return res;
}

inline double sum(const Mat& mat)
{
    double sum = 0;
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            sum += MAT_AT(mat, i, j);
        }
    }
    return sum;
}

inline Mat add_scalar(const Mat& mat1, double scalar)
{
    const Mat res = allocate_mat(mat1.rows, mat1.cols);

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            MAT_AT(res, i, j) = MAT_AT(mat1, i, j) + scalar;
        }
    }

    return res;
}

inline Mat mult_scalar(const Mat& mat1, double scalar)
{
    const Mat res = allocate_mat(mat1.rows, mat1.cols);

    for (int i = 0; i < mat1.rows; ++i)
    {
        for (int j = 0; j < mat1.cols; ++j)
        {
            MAT_AT(res, i, j) = MAT_AT(mat1, i, j) * scalar;
        }
    }

    return res;
}

inline Mat transpose(const Mat& mat)
{
    Mat transposed = allocate_mat(mat.cols, mat.rows);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(transposed, j, i) = MAT_AT(mat, i, j);
        }
    }

    return transposed;
}

Mat forward_pass(const Weights& weights, const Traces& traces, const Mat& X)
{
    assert(weights.count == traces.count);
    traces.items[0] = X;

    Mat prev = dot(weights.items[0], X);
    for (int l = 1; l < weights.count; ++l)
    {
        Mat new_step = dot(weights.items[l], prev);
        traces.items[l] = prev;
        prev = new_step;
    }

    return prev;
}

double mse_loss(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows && logits.rows == 1);
    double mse = 0;
    for (int i = 0; i < logits.cols; ++i)
    {
        mse += (MAT_AT(logits, 0, i) - MAT_AT(y, 0, i)) * (MAT_AT(logits, 0, i) - MAT_AT(y, 0, i));
    }

    return mse / logits.cols;
}

Mat mse_der(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows && logits.rows == 1);
    Mat ders = allocate_mat(y.rows, y.cols);

    for (int i = 0; i < logits.cols; ++i)
    {
        MAT_AT(ders, 0, i) = 2 * (MAT_AT(logits, 0, i) - MAT_AT(y, 0, i)) / logits.cols;
    }
    return ders;
}

Weights backward_pass(const Weights& weights, const Traces& traces, const Mat& loss_der)
{
    Weights grads;
    grads.count = weights.count;
    grads.items = (Mat*)malloc(sizeof(Mat) * grads.count);

    Mat upstream = loss_der;

    for (int l = static_cast<int>(grads.count) - 1; l >= 0; --l)
    {
        grads.items[l] = dot(upstream, transpose(traces.items[l]));
        upstream = dot(transpose(weights.items[l]), upstream);
    }

    return grads;
}

void step_optimizer(Weights weights, const Weights& grads, const double& lr)
{
    for (int layer = 0; layer < weights.count; ++layer)
    {
        Mat scaled_grads = mult_scalar(grads.items[layer], -lr);
        Mat new_wei = add(weights.items[layer], scaled_grads);
        MAT_FREE(weights.items[layer]);
        weights.items[layer] = new_wei;
        MAT_FREE(scaled_grads);
    }
}

int main()
{
    // Dataset (simple y = 2*x1 + x2 function)
    Mat X = allocate_mat(2, 5);
    MAT_AT(X, 0, 0) = 1;
    MAT_AT(X, 1, 0) = 1;
    MAT_AT(X, 0, 1) = 2;
    MAT_AT(X, 1, 1) = 1;
    MAT_AT(X, 0, 2) = 3;
    MAT_AT(X, 1, 2) = 1;
    MAT_AT(X, 0, 3) = 4;
    MAT_AT(X, 1, 3) = 1;
    MAT_AT(X, 0, 4) = 5;
    MAT_AT(X, 1, 4) = 1;


    Mat y = allocate_mat(1, 5);
    MAT_AT(y, 0, 0) = 1 * 2 + 1;
    MAT_AT(y, 0, 1) = 2 * 2 + 1;
    MAT_AT(y, 0, 2) = 3 * 2 + 1;
    MAT_AT(y, 0, 3) = 4 * 2 + 1;
    MAT_AT(y, 0, 4) = 5 * 2 + 1;

    // XOR dataset
    // Mat X = allocate_mat(2, 4);
    // Mat y = allocate_mat(1, 4);
    // for (int i = 0; i < 2; ++i)
    // {
    //     for (int j = 0; j < 2; ++j)
    //     {
    //         int ix = 2 * i + j;
    //         MAT_AT(X, 0, ix) = i;
    //         MAT_AT(X, 1, ix) = j;
    //         MAT_AT(y, 0, ix) = i ^ j;
    //     }
    // }

    std::cout << "X:" << std::endl;
    // X = transpose(X);
    print_mat(X);
    std::cout << "y:" << std::endl;
    print_mat(y);

    double lr = 0.01;
    int net[] = {2, 2, 1};

    Weights weights;
    weights.items = (Mat*)malloc(sizeof(Mat) * std::size(net));
    weights.count = std::size(net) - 1;

    for (int i = 0; i < std::size(net) - 1; ++i)
    {
        weights.items[i] = allocate_mat(net[i+1], net[i]);
        fill_mat(weights.items[i], 0.1);
    }

    for (int step = 0; step < 50; ++step)
    {
        Traces traces;
        traces.count = weights.count;
        traces.items = static_cast<Mat*>(malloc(sizeof(Mat) * traces.count));

        Mat logits = forward_pass(weights, traces, X);
        double loss = mse_loss(logits, y);
        Mat loss_der = mse_der(logits, y);

        std::cout << "Loss:            " << loss << std::endl;
        // std::cout << "Loss derivative: " << std::endl;

        Weights grads = backward_pass(weights, traces, loss_der);
        step_optimizer(weights, grads, lr);

        std::cout << "===================================" << std::endl;

        MAT_FREE(logits);
        MAT_FREE(loss_der);
        // MAT_FREE(grads);
        // TODO free traces
    }

    print_mat(weights.items[0]);

    for (size_t w = 0; w < weights.count; ++w)
    {
        MAT_FREE(weights.items[w]);
    }
    free(weights.items);

    MAT_FREE(X);
    MAT_FREE(y);
    return 0;
}