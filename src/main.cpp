#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>



#define MAT_AT(mat, i, j) static_cast<double*>(GC_AT((mat).values))[assert(i < (mat).rows && j < (mat).cols && i >= 0 && j >= 0), (i) * (mat).cols + (j)]

#define GC_AT(ptr) static_cast<void*>(gc_mem + ptr)
#define GC_DELETE(ptr) reinterpret_cast<GCObject*>(gc_mem + (ptr) - sizeof(GCObject))->alive = false

char* gc_mem;
size_t gc_mem_size;

typedef size_t GCPtr;
typedef struct
{
    bool alive;
    size_t size;
    GCPtr next_offset;
    GCPtr my_offset;
} GCObject;

void gc_init(size_t capacity)
{
    gc_mem = static_cast<char*>(malloc(capacity));
    if (gc_mem == nullptr) abort();
    memset(gc_mem, 0, capacity);
    gc_mem_size = capacity;

    auto *current = reinterpret_cast<GCObject*>(gc_mem);
    current->alive = true;
    current->size = sizeof(GCObject);
    current->next_offset = 0;
    current->my_offset = 0;
}

void gc_free()
{
    free(gc_mem);
    gc_mem_size = 0;
}

GCPtr gc_alloc(size_t size)
{
    auto *current = reinterpret_cast<GCObject*>(gc_mem);
    while (current->next_offset > 0)
    {
        GCPtr next = current->next_offset;
        GCPtr end_of_current = current->my_offset + current->size;
        if (next - end_of_current >= size + sizeof(GCObject))
        {
            current->next_offset = end_of_current;
            auto new_object = reinterpret_cast<GCObject*>(gc_mem + end_of_current);
            new_object->alive = true;
            new_object->size = size + sizeof(GCObject);
            new_object->next_offset = next;
            new_object->my_offset = end_of_current;
            return new_object->my_offset + sizeof(GCObject);
        }

        current = reinterpret_cast<GCObject*>(gc_mem + next);
    }

    if (current->my_offset + current->size + size + sizeof(GCObject) > gc_mem_size) abort();
    current->next_offset = current->my_offset + current->size;
    auto *new_object = reinterpret_cast<GCObject*>(gc_mem + current->next_offset);
    new_object->alive = true;
    new_object->size = size + sizeof(GCObject);
    new_object->next_offset = 0;
    new_object->my_offset = current->next_offset;
    return new_object->my_offset + sizeof(GCObject);
}

GCPtr gc_temp_alloc(size_t size)
{
    GCPtr p = gc_alloc(size);
    GC_DELETE(p);
    return p;
}

void gc_cleanup()
{
    auto *current = reinterpret_cast<GCObject*>(gc_mem);
    if (!current->alive) abort(); // First one cannot be killed

    GCPtr prev = 0;
    while (current->next_offset > 0)
    {
        current = reinterpret_cast<GCObject*>(gc_mem + current->next_offset);
        if (!current->alive)
        {
            // std::cout << "Found 1 stale reference at " << current->my_offset << std::endl;
            reinterpret_cast<GCObject*>(gc_mem + prev)->next_offset = current->next_offset;
            current = reinterpret_cast<GCObject*>(gc_mem + prev);
            continue;
        }

        prev = current->my_offset;
    }
}

typedef struct
{
    int rows, cols;
    GCPtr values;
} Mat;

typedef struct
{
    size_t count;
    Mat *items;
} Weights;

typedef struct
{
    size_t count;
    Mat *layers;
    Mat *activations;
} Traces;

inline Mat allocate_mat(int rows, int cols)
{
    Mat mat = {
        .rows=rows,
        .cols=cols,
        .values=gc_alloc(sizeof(double) * rows * cols),
    };
    return mat;
}

inline Mat allocate_temp_mat(int rows, int cols)
{
    Mat mat = {
        .rows=rows,
        .cols=cols,
        .values=gc_temp_alloc(sizeof(double) * rows * cols),
    };
    return mat;
}

inline Mat mat_temp_dup(const Mat& mat)
{
    Mat res = allocate_temp_mat(mat.rows, mat.cols);

    auto src = static_cast<double*>(GC_AT(mat.values));
    auto dst = static_cast<double*>(GC_AT(res.values));

    memcpy(dst, src, sizeof(double) * mat.rows * mat.cols);
    return res;
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

inline void fill_random_mat(const Mat &mat, double value)
{
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            double r = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            MAT_AT(mat, i, j) = (r * 2.0 - 1.0) * value;
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
    const Mat res = allocate_temp_mat(mat1.rows, mat2.cols);

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
    const Mat res = allocate_temp_mat(mat1.rows, mat1.cols);

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
    const Mat res = allocate_temp_mat(mat1.rows, mat1.cols);

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
    const Mat res = allocate_temp_mat(mat1.rows, mat1.cols);

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
    const Mat res = allocate_temp_mat(mat1.rows, mat1.cols);

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
    Mat transposed = allocate_temp_mat(mat.cols, mat.rows);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(transposed, j, i) = MAT_AT(mat, i, j);
        }
    }

    return transposed;
}

Mat relu(const Mat& mat)
{
    Mat result = allocate_temp_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            if (MAT_AT(mat, i, j) < 0)
            {
                MAT_AT(result, i, j) = 0;
            }
            else
            {
                MAT_AT(result, i, j) = MAT_AT(mat, i, j);
            }
        }
    }

    return result;
}

Mat relu_der(const Mat& mat)
{
    Mat result = allocate_temp_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            if (MAT_AT(mat, i, j) < 0)
            {
                MAT_AT(result, i, j) = 0;
            } else
            {
                MAT_AT(result, i, j) = 1;
            }
        }
    }

    return result;
}

Mat forward_pass(const Weights& weights, const Traces& traces, const Mat& X)
{
    assert(weights.count == traces.count);
    traces.layers[0] = X;

    Mat prev = dot(weights.items[0], X);
    traces.activations[0] = mat_temp_dup(prev);
    prev = relu(prev);
    for (int l = 1; l < weights.count; ++l)
    {
        Mat new_step = dot(weights.items[l], prev);
        traces.activations[l] = mat_temp_dup(new_step);

        new_step = relu(new_step);
        traces.layers[l] = prev;

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
    Mat ders = allocate_temp_mat(y.rows, y.cols);

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
        upstream = mult(upstream, relu_der(traces.activations[l]));
        Mat grad = dot(upstream, transpose(traces.layers[l]));
        grads.items[l] = grad;
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
        weights.items[layer] = new_wei;
    }
}

int main()
{
    gc_init(1024 * 1024);

    // Dataset (simple y = 2*x1 + x2 function)
    // Mat X = allocate_mat(2, 10 * 10);
    // Mat y = allocate_mat(1, 10 * 10);
    // int ix = 0;
    // for (int x1 = 1; x1 <= 10; x1++)
    // {
    //     for (int x2 = 1; x2 <= 10; x2++)
    //     {
    //         MAT_AT(X, 0, ix) = x1;
    //         MAT_AT(X, 1, ix) = x2;
    //         MAT_AT(y, 0, ix) = 1.7 * x1 + x2;
    //         ix++;
    //     }
    // }

    // XOR dataset
    Mat X = allocate_mat(2, 4);
    Mat y = allocate_mat(1, 4);
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            int ix = 2 * i + j;
            MAT_AT(X, 0, ix) = i;
            MAT_AT(X, 1, ix) = j;
            MAT_AT(y, 0, ix) = i ^ j;
        }
    }

    std::cout << "X:" << std::endl;
    print_mat(X);
    std::cout << "y:" << std::endl;
    print_mat(y);

    double lr = 0.001;
    int net[] = {2, 2, 1};

    Weights weights;
    weights.items = (Mat*)malloc(sizeof(Mat) * std::size(net));
    weights.count = std::size(net) - 1;

    for (int i = 0; i < std::size(net) - 1; ++i)
    {
        weights.items[i] = allocate_mat(net[i+1], net[i]);
        fill_random_mat(weights.items[i], 0.1);
    }

    for (int step = 0; step < 50; ++step)
    {
        Traces traces;
        traces.count = weights.count;
        traces.layers = static_cast<Mat*>(malloc(sizeof(Mat) * traces.count));
        traces.activations = static_cast<Mat*>(malloc(sizeof(Mat) * traces.count));

        Mat logits = forward_pass(weights, traces, X);
        double loss = mse_loss(logits, y);
        Mat loss_der = mse_der(logits, y);

        std::cout << "Loss:            " << loss << std::endl;
        Weights grads = backward_pass(weights, traces, loss_der);
        step_optimizer(weights, grads, lr);

        std::cout << "===================================" << std::endl;
        gc_cleanup();
    }


    print_mat(weights.items[0]);

    for (size_t w = 0; w < weights.count; ++w)
    {
        GC_DELETE(weights.items[w].values);
    }
    free(weights.items);

    GC_DELETE(X.values);
    GC_DELETE(y.values);

    gc_free();
    return 0;
}
