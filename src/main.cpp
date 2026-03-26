#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>
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

enum LayerKind
{
    LYR_LINEAR,
    LYR_SIGMOID,
    LYR_RELU,
};

typedef struct
{
    int in_features;
    int out_features;
    LayerKind kind;
} LayerSpec;

typedef struct
{
    size_t count;
    LayerSpec *specs;
    Mat *weights;
    Mat *biases;
} Model;

typedef struct
{
    size_t count;
    Mat *weights;
    Mat *biases;
} Grads;

typedef struct
{
    size_t count;
    Mat *inputs;
    Mat *outputs;
} Traces;

inline Mat null_mat()
{
    Mat mat = {
        .rows=0,
        .cols=0,
        .values=0,
    };
    return mat;
}

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

inline Mat add_bias(const Mat& mat, const Mat& bias)
{
    assert(mat.rows == bias.rows);
    assert(bias.cols == 1);
    const Mat res = allocate_temp_mat(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(res, i, j) = MAT_AT(mat, i, j) + MAT_AT(bias, i, 0);
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

inline Mat sum_cols(const Mat& mat)
{
    Mat res = allocate_temp_mat(mat.rows, 1);
    for (int i = 0; i < mat.rows; ++i)
    {
        double row_sum = 0;
        for (int j = 0; j < mat.cols; ++j)
        {
            row_sum += MAT_AT(mat, i, j);
        }
        MAT_AT(res, i, 0) = row_sum;
    }

    return res;
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

Mat sigmoid(const Mat& mat)
{
    Mat result = allocate_temp_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(result, i, j) = 1.0 / (1.0 + exp(-MAT_AT(mat, i, j)));
        }
    }

    return result;
}

Mat sigmoid_der_from_output(const Mat& output)
{
    Mat result = allocate_temp_mat(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i)
    {
        for (int j = 0; j < output.cols; ++j)
        {
            double s = MAT_AT(output, i, j);
            MAT_AT(result, i, j) = s * (1.0 - s);
        }
    }

    return result;
}

Mat relu(const Mat& mat)
{
    Mat result = allocate_temp_mat(mat.rows, mat.cols);
    for (int i = 0; i < mat.rows; ++i)
    {
        for (int j = 0; j < mat.cols; ++j)
        {
            MAT_AT(result, i, j) = std::max(0.0, MAT_AT(mat, i, j));
        }
    }

    return result;
}

Mat relu_der_from_output(const Mat& output)
{
    Mat result = allocate_temp_mat(output.rows, output.cols);
    for (int i = 0; i < output.rows; ++i)
    {
        for (int j = 0; j < output.cols; ++j)
        {
            MAT_AT(result, i, j) = MAT_AT(output, i, j) > 0.0 ? 1.0 : 0.0;
        }
    }

    return result;
}

Model create_model(const LayerSpec* specs, size_t count)
{
    Model model;
    model.count = count;
    model.specs = static_cast<LayerSpec*>(malloc(sizeof(LayerSpec) * count));
    model.weights = static_cast<Mat*>(malloc(sizeof(Mat) * count));
    model.biases = static_cast<Mat*>(malloc(sizeof(Mat) * count));

    memcpy(model.specs, specs, sizeof(LayerSpec) * count);

    int current_features = specs[0].in_features;
    for (size_t l = 0; l < count; ++l)
    {
        assert(model.specs[l].in_features == current_features);

        model.weights[l] = null_mat();
        model.biases[l] = null_mat();

        switch (model.specs[l].kind)
        {
            case LYR_LINEAR:
                model.weights[l] = allocate_mat(model.specs[l].out_features, current_features);
                fill_random_mat(model.weights[l], 1.0);
                model.biases[l] = allocate_mat(model.specs[l].out_features, 1);
                fill_random_mat(model.biases[l], 1.0);
                break;
            case LYR_RELU:
            case LYR_SIGMOID:
                assert(model.specs[l].out_features == current_features);
                break;
            default:
                abort();
        }

        current_features = model.specs[l].out_features;
    }

    return model;
}

void free_model(const Model& model)
{
    for (size_t l = 0; l < model.count; ++l)
    {
        if (model.specs[l].kind == LYR_LINEAR)
        {
            GC_DELETE(model.weights[l].values);
            GC_DELETE(model.biases[l].values);
        }
    }

    free(model.specs);
    free(model.weights);
    free(model.biases);
}

Mat forward_pass(const Model& model, const Traces& traces, const Mat& X)
{
    assert(model.count == traces.count);

    Mat prev = X;
    for (size_t l = 0; l < model.count; ++l)
    {
        traces.inputs[l] = prev;

        switch (model.specs[l].kind)
        {
            case LYR_LINEAR:
                prev = add_bias(dot(model.weights[l], prev), model.biases[l]);
                break;
            case LYR_SIGMOID:
                prev = sigmoid(prev);
                break;
            case LYR_RELU:
                prev = relu(prev);
                break;
            default:
                abort();
        }

        traces.outputs[l] = mat_temp_dup(prev);
    }

    return prev;
}

double bce_loss(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows && logits.rows == 1);
    double bce = 0;
    for (int i = 0; i < logits.cols; ++i)
    {
        double pred = MAT_AT(logits, 0, i);
        pred = std::fmax(1e-7, std::fmin(1.0 - 1e-7, pred));
        double target = MAT_AT(y, 0, i);
        bce += -(target * log(pred) + (1.0 - target) * log(1.0 - pred));
    }

    return bce / logits.cols;
}

double mse_loss(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows);

    double mse = 0;
    for (int i = 0; i < logits.rows; ++i)
    {
        for (int j = 0; j < logits.cols; ++j)
        {
            double diff = MAT_AT(logits, i, j) - MAT_AT(y, i, j);
            mse += diff * diff;
        }
    }

    return mse / (logits.rows * logits.cols);
}

Mat bce_der(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows && logits.rows == 1);
    Mat ders = allocate_temp_mat(y.rows, y.cols);

    for (int i = 0; i < logits.cols; ++i)
    {
        double pred = MAT_AT(logits, 0, i);
        pred = std::fmax(1e-7, std::fmin(1.0 - 1e-7, pred));
        double target = MAT_AT(y, 0, i);
        MAT_AT(ders, 0, i) = (pred - target) / (pred * (1.0 - pred) * logits.cols);
    }
    return ders;
}

Mat mse_der(const Mat& logits, const Mat& y)
{
    assert(logits.cols == y.cols);
    assert(logits.rows == y.rows);

    Mat ders = allocate_temp_mat(y.rows, y.cols);
    double scale = 2.0 / (logits.rows * logits.cols);

    for (int i = 0; i < logits.rows; ++i)
    {
        for (int j = 0; j < logits.cols; ++j)
        {
            MAT_AT(ders, i, j) = scale * (MAT_AT(logits, i, j) - MAT_AT(y, i, j));
        }
    }

    return ders;
}

Grads backward_pass(const Model& model, const Traces& traces, const Mat& loss_der)
{
    Grads grads;
    grads.count = model.count;
    grads.weights = static_cast<Mat*>(malloc(sizeof(Mat) * grads.count));
    grads.biases = static_cast<Mat*>(malloc(sizeof(Mat) * grads.count));

    for (size_t l = 0; l < grads.count; ++l)
    {
        grads.weights[l] = null_mat();
        grads.biases[l] = null_mat();
    }

    Mat upstream = loss_der;

    for (int l = static_cast<int>(model.count) - 1; l >= 0; --l)
    {
        switch (model.specs[l].kind)
        {
            case LYR_SIGMOID:
                upstream = mult(upstream, sigmoid_der_from_output(traces.outputs[l]));
                break;
            case LYR_RELU:
                upstream = mult(upstream, relu_der_from_output(traces.outputs[l]));
                break;
            case LYR_LINEAR:
                grads.weights[l] = dot(upstream, transpose(traces.inputs[l]));
                grads.biases[l] = sum_cols(upstream);
                upstream = dot(transpose(model.weights[l]), upstream);
                break;
            default:
                abort();
        }
    }

    return grads;
}

void step_optimizer(const Model& model, const Grads& grads, const double& lr)
{
    for (int layer = 0; layer < static_cast<int>(model.count); ++layer)
    {
        if (model.specs[layer].kind != LYR_LINEAR)
        {
            continue;
        }

        for (int i = 0; i < model.weights[layer].rows; ++i)
        {
            for (int j = 0; j < model.weights[layer].cols; ++j)
            {
                MAT_AT(model.weights[layer], i, j) -= lr * MAT_AT(grads.weights[layer], i, j);
            }
        }

        for (int i = 0; i < model.biases[layer].rows; ++i)
        {
            MAT_AT(model.biases[layer], i, 0) -= lr * MAT_AT(grads.biases[layer], i, 0);
        }
    }
}

int main()
{
    gc_init(1024 * 1024);
    srand(0);

    // Dataset (simple y = 2*x1 + x2 function)
    Mat X = allocate_mat(2, 10 * 10);
    Mat y = allocate_mat(1, 10 * 10);
    int ix = 0;
    for (int x1 = 1; x1 <= 10; x1++)
    {
        for (int x2 = 1; x2 <= 10; x2++)
        {
            MAT_AT(X, 0, ix) = x1;
            MAT_AT(X, 1, ix) = x2;
            MAT_AT(y, 0, ix) = 1.7 * x1 + x2;
            ix++;
        }
    }

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
    print_mat(X);
    std::cout << "y:" << std::endl;
    print_mat(y);

    double lr = 0.01;
    LayerSpec specs[] = {
        {2, 3, LYR_LINEAR},
        {3, 3, LYR_SIGMOID},
        {3, 3, LYR_LINEAR},
        {3, 3, LYR_SIGMOID},
        {3, 1, LYR_LINEAR},
    };

    Model model = create_model(specs, std::size(specs));

    for (int step = 0; step < 100000; ++step)
    {
        Traces traces;
        traces.count = model.count;
        traces.inputs = static_cast<Mat*>(malloc(sizeof(Mat) * traces.count));
        traces.outputs = static_cast<Mat*>(malloc(sizeof(Mat) * traces.count));

        Mat logits = forward_pass(model, traces, X);
        double loss = mse_loss(logits, y);
        Mat loss_der = mse_der(logits, y);

        if (step % 5000 == 0 || step == 99999)
        {
            std::cout << "Step " << step << " Loss: " << loss << std::endl;
        }
        Grads grads = backward_pass(model, traces, loss_der);
        step_optimizer(model, grads, lr);

        free(grads.weights);
        free(grads.biases);
        free(traces.inputs);
        free(traces.outputs);
        gc_cleanup();
    }

    Traces final_traces;
    final_traces.count = model.count;
    final_traces.inputs = static_cast<Mat*>(malloc(sizeof(Mat) * model.count));
    final_traces.outputs = static_cast<Mat*>(malloc(sizeof(Mat) * model.count));
    Mat logits = forward_pass(model, final_traces, X);
    std::cout << "Predictions:" << std::endl;
    print_mat(logits);
    free(final_traces.inputs);
    free(final_traces.outputs);

    free_model(model);

    GC_DELETE(X.values);
    GC_DELETE(y.values);

    gc_free();
    return 0;
}
