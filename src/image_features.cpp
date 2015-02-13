#include "image_features.h"

#include <vector>
#include <tuple>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

typedef Matrix<tuple<float, float>> PolarGradientMatrix;
typedef Matrix<tuple<uint, uint, uint>> RGBMatrix;

#ifndef M_PI
#define M_PI 3.14159265;
#endif

const double eps = 1e-7;


static inline uint
grayscale(uint red, uint green, uint blue)
{
    //magical constants for great justice! see task2.pdf, p.4
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

//allocates dynamic memory!
static inline Matrix<uint> *
NewGrayscaleMatrixFromBMP(BMP *img)
{
    auto res = new Matrix<uint>(img->TellHeight(), img->TellWidth());
    
    for (uint i = 0; i < res->n_rows; ++i) {
        for (uint j = 0; j < res->n_cols; ++j) {
            RGBApixel pixel = img->GetPixel(j, i);
            (*res)(i, j) = grayscale(pixel.Red, pixel.Blue, pixel.Green);
        }
    }
    
    return res;
}

//allocates dynamic memory!
static inline RGBMatrix *
NewRGBMatrixFromBMP(BMP *img)
{
    auto res = new RGBMatrix(img->TellHeight(), img->TellWidth());

    for (uint i = 0; i < res->n_rows; ++i) {
        for (uint j = 0; j < res->n_cols; ++j) {
            RGBApixel pixel = img->GetPixel(j, i);
            (*res)(i, j) = make_tuple(pixel.Red, pixel.Green, pixel.Blue);
        }
    }

    return res;
}

class PolarGradient
{
public:
    // Radiuses of neighbourhoud, which are passed to that operator
    static const int vert_radius = 1, hor_radius = 1;
    
    tuple<float, float>
    operator() (const Matrix<uint> &grayscale_matrix) const
    {
        //dimensions(grayscale_matrix) = {2, 2} due to radius = 1
        //horyzontal Sobel filter = (grayscale_matrix[:][1] x {-1, 0, 1}):
        uint grad_x = grayscale_matrix(2, 1) - grayscale_matrix(0, 1);
        //vertical Sobel filter = (grayscale_matrix[1][:] x {1, 0, -1}*):
        uint grad_y = grayscale_matrix(1, 0) - grayscale_matrix(1, 2);
        float r = sqrt(grad_x*grad_x + grad_y*grad_y);
        float phi = atan2(grad_y, grad_x);
        return make_tuple(r, phi);
    }
};

class HOGDescriptor
{
public:
    const uint row_split_c, col_split_c;
    
    HOGDescriptor(uint _row_split_c, uint _col_split_c)
        : row_split_c(_row_split_c), col_split_c(_col_split_c)
    {}
    
    static inline uint
    grad_dir_part(double phi, double grad_dir_split_c)
    {
        return floor(grad_dir_split_c * (phi + M_PI) / (2 * M_PI));
    }
    
    void
    operator() (
        const PolarGradientMatrix &pol_grad_matr,
        vector<float> *HOG
        ) const
    {
        const uint grad_dir_split_c = 16;
        double r, phi;
        const uint HOG_len = HOG->size() + grad_dir_split_c;
        HOG->resize(HOG_len);
        for (uint i = 0; i < pol_grad_matr.n_rows; ++i) {
            for (uint j = 0; j < pol_grad_matr.n_cols; ++j) {
                tie(r, phi) = pol_grad_matr(i, j);
                (*HOG)[HOG_len - grad_dir_part(phi, grad_dir_split_c)] += r;
            }
        }
    
        double sum = 0.0;
        for (int i = grad_dir_split_c; i > 0; --i) {
            sum += (*HOG)[HOG_len - i] * (*HOG)[HOG_len - i];
        }
        double norm = sqrt(sum);
        if (norm > eps) {
            for (int i = grad_dir_split_c; i > 0; --i) {
                (*HOG)[HOG_len - i] /= norm;
            }
        }
    }
};

class ColourDescriptor
{
public:
    const uint row_split_c, col_split_c;
    
    ColourDescriptor(uint _row_split_c, uint _col_split_c)
        : row_split_c(_row_split_c), col_split_c(_col_split_c)
    {}
    
    void
    operator() (const RGBMatrix &rgb_cell, vector<float> *features) const
    {
        uint r_sum = 0, g_sum = 0, b_sum = 0, r, g, b;
        for (uint i = 0; i < rgb_cell.n_rows; ++i) {
            for (uint j = 0; j < rgb_cell.n_cols; ++j) {
                tie(r, g, b) = rgb_cell(i, j);
                r_sum += r;
                g_sum += g;
                b_sum += b;
            }
        }
        uint pixels_count = rgb_cell.n_rows * rgb_cell.n_cols;
        float r_avg = r_sum / (255.0 * pixels_count);
        features->push_back(r_avg);
        float g_avg = g_sum / (255.0 * pixels_count);
        features->push_back(g_avg);
        float b_avg = b_sum / (255.0 * pixels_count);
        features->push_back(b_avg);
    }
};
                

static inline void
BuildHOGPyramid(const PolarGradientMatrix &pol_grad_matr, vector<float> *HOG)
{
    const uint rows_in_cell = 8, cols_in_cell = 8;
    const uint cells_row_c = 2, cells_col_c = 2;
    pol_grad_matr.cell_map(
        HOGDescriptor(rows_in_cell, cols_in_cell),   //entire image as one cell
        HOG
    );
    pol_grad_matr.cell_map(
        HOGDescriptor(rows_in_cell * cells_row_c, cols_in_cell * cells_col_c),
        HOG
    );
}

static inline void
AddColourFeatures(const RGBMatrix &rgb_matrix, vector<float> *features)
{
    const uint rows_split_c = 8, cols_split_c = 8;
    rgb_matrix.cell_map(ColourDescriptor(rows_split_c, cols_split_c), features);
}


static inline void
square_core_transform(float z, float lambda, vector<float> *results)
{
    if (z < eps) {
        z = eps;
    }
    float r = sqrt(z / cosh(M_PI * lambda));
    float phi = lambda * log(z);
    //transform to cartesian coordinates (SVM works faster with them):
    float x = r * cos(phi);
    float y = r * sin(phi);
    results->push_back(x);
    results->push_back(y);
}

static inline void
NonLinearTransform(vector<float> old_features, vector<float> *new_features)
{
    const float L = 0.32;
    for (auto &feature : old_features) {
        square_core_transform(feature, -L, new_features);
        square_core_transform(feature, 0.0, new_features);
        square_core_transform(feature, L, new_features);
    }
}

void
ExtractFeatures(const TDataSet &data_set, TFeatures *features)
{
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        Matrix<uint> *grayscale_matrix =
            NewGrayscaleMatrixFromBMP(data_set[image_idx].first);
        PolarGradientMatrix grad_matr =
            grayscale_matrix->unary_map(PolarGradient());
        RGBMatrix *rgb_matrix = 
            NewRGBMatrixFromBMP(data_set[image_idx].first);
        
        vector<float> cur_image_features_lin, cur_image_features_nonlin;
        BuildHOGPyramid(grad_matr, &cur_image_features_lin);
        AddColourFeatures(*rgb_matrix, &cur_image_features_lin);
        NonLinearTransform(cur_image_features_lin, &cur_image_features_nonlin);
        
        delete grayscale_matrix;
        delete rgb_matrix;
        
        features->push_back(
            make_pair(cur_image_features_nonlin, data_set[image_idx].second)
        );
    }
}
