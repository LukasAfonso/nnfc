#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>

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
    size_t capacity;
    Mat *items;
} Weights;

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
    assert(mat1.rows == mat2.rows);
    assert(mat1.cols == mat2.cols);
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

Mat forward_pass(const Weights& weights, const Mat& X)
{
    // No bias
    // 1 Input 1 Output
    assert(weights.count == 1);
    Mat wei = weights.items[0];

    assert(wei.cols == X.rows);
    return dot(wei, X);
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

Mat backward_pass(const Weights& weights, const Mat& loss_der, const Mat& X)
{
    // No bias
    // 1 Input 1 Output
    assert(weights.count == 1);
    assert(weights.items[0].rows == weights.items[0].cols && weights.items[0].rows == 1);

    Mat grads = allocate_mat(weights.items[0].rows, weights.items[0].cols);
    fill_mat(grads, 0.0);
    for (int i = 0; i < X.cols; ++i)
    {
        MAT_AT(grads, 0, 0) += MAT_AT(loss_der, 0, i) * MAT_AT(X, 0, i);
    }
    return grads;
}

void step_optimizer(Weights weights, const Mat& grads, const double& lr)
{
    Mat scaled_grads = mult_scalar(grads, -lr);
    for (size_t w = 0; w < weights.count; ++w)
    {
        Mat new_wei = add(weights.items[w], scaled_grads);
        MAT_FREE(weights.items[w]);
        weights.items[w] = new_wei;
    }

    MAT_FREE(scaled_grads);
}

int main()
{
    // Dataset (simple y = 2x function)
    Mat X = allocate_mat(1, 5);
    MAT_AT(X, 0, 0) = 1;
    MAT_AT(X, 0, 1) = 2;
    MAT_AT(X, 0, 2) = 3;
    MAT_AT(X, 0, 3) = 4;
    MAT_AT(X, 0, 4) = 5;
    std::cout << "X:" << std::endl;
    print_mat(X);

    Mat y = allocate_mat(1, 5);
    MAT_AT(y, 0, 0) = 1 * 2.0;
    MAT_AT(y, 0, 1) = 2 * 2.0;
    MAT_AT(y, 0, 2) = 3 * 2.0;
    MAT_AT(y, 0, 3) = 4 * 2.0;
    MAT_AT(y, 0, 4) = 5 * 2.0;
    std::cout << "y:" << std::endl;
    print_mat(y);

    double lr = 0.02;
    int depth = 1;
    int n_input = 1;
    int n_output = 1;

    Weights weights;
    weights.items = (Mat*)malloc(sizeof(Mat) * depth);
    weights.count = depth;
    weights.capacity = depth;

    weights.items[0] = allocate_mat(n_output, n_input);
    fill_mat(weights.items[0], 0.0);

    for (int step = 0; step < 5; ++step)
    {
        Mat logits = forward_pass(weights, X);
        double loss = mse_loss(logits, y);
        Mat loss_der = mse_der(logits, y);

        std::cout << "Loss:            " << loss << std::endl;
        std::cout << "Loss derivative: " << std::endl;
        print_mat(loss_der);

        Mat grads = backward_pass(weights, loss_der, X);
        print_mat(grads);
        step_optimizer(weights, grads, lr);

        std::cout << "===================================" << std::endl;

        MAT_FREE(logits);
        MAT_FREE(loss_der);
        MAT_FREE(grads);
    }

    for (size_t w = 0; w < weights.count; ++w)
    {
        MAT_FREE(weights.items[w]);
    }
    free(weights.items);

    MAT_FREE(X);
    MAT_FREE(y);
    return 0;
}