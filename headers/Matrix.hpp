#pragma once

#include <utility>
#include <stdexcept>

class Matrix
{
public:
    using Rows = unsigned int;
    using Cols = unsigned int;
    using MatrixSize = std::pair<Rows, Cols>;

    Matrix(unsigned int rows, unsigned int cols, double default_values = 0)
    {
        _resize(rows, cols, default_values);
    }

    Matrix(const std::vector<double>& lhl)
    {
        _resize(lhl.size(), 1);
        
        for (int i = 0; i < lhl.size(); i++)
        {
            this->operator()(i, 0) = lhl.at(i);
        }
    }

    Matrix(const Matrix& lhl)
    {
        const auto lhl_size = lhl.size();
        _resize(lhl_size.first, lhl_size.second);
        
        for (int i = 0; i < lhl_size.first; i++)
        {
            for (int j = 0; j < lhl_size.second; j++)
            {
                this->operator()(i, j) = lhl(i, j);
            }
        }
    }
    
    ~Matrix()
    {
        // if (m_matrix != nullptr)
        // {
        //     for (int i = 0; i < m_rows; i++)
        //         delete m_matrix[i];
        //     delete m_matrix;
        // }
    }

    MatrixSize size() const
    {
        return {m_rows, m_cols};
    }

    static Matrix transponate(const Matrix& lhl)
    {
        const auto orig_size = lhl.size();
        Matrix result(orig_size.second, orig_size.first);

        for (auto i = 0; i < orig_size.first; i++)
        {
            for (auto j = 0; j < orig_size.second; j++)
            {
                result(j, i) = lhl(i, j);
            }
        }

        return result;
    }

    double& operator()(unsigned int i, unsigned int j)
    {
        if (i >= m_rows or j >= m_cols)
            throw std::runtime_error("Matrix::operator() out of bounds.");

        return m_matrix[i][j];
    }

    const double& operator()(unsigned int i, unsigned int j) const
    {
        if (i >= m_rows or j >= m_cols)
            throw std::runtime_error("Matrix::operator() out of bounds.");

        return m_matrix[i][j];
    }

    Matrix operator*(Matrix& lhl)
    {
        if (m_cols != lhl.size().first)
            throw std::runtime_error("Matrix::operator*() Matrixes are not compatible.");

        Matrix result(m_rows, lhl.size().second);
        for (int i = 0; i < result.size().first; i++)
        {
            for (int j = 0; j < result.size().second; j++)
            {
                double sum = 0;
                for(int k = 0; k < m_cols; k++)
                {
                    auto a = m_matrix[i][k];
			        auto b = lhl(k, j); 
                    sum += m_matrix[i][k] * lhl(k, j);
                }
                result(i, j) = sum;
            }
        }

        return result;
    }

    Matrix operator*(double lhl)
    {
        Matrix result(m_rows, m_cols);
        for (int i = 0; i < result.size().first; i++)
        {
            for (int j = 0; j < result.size().second; j++)
            {
                result(i, j) = m_matrix[i][j] * lhl;
            }
        }

        return result;
    }

    Matrix operator=(const std::vector<double>& lhl)
    {
        _resize(lhl.size(), 1);
        
        for (int i = 0; i < lhl.size(); i++)
        {
            this->operator()(i, 0) = lhl.at(i);
        }

        return *this;
    }

    Matrix operator=(const Matrix& lhl)
    {
        _resize(lhl.size().first, lhl.size().second);
        
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_cols; j++)
            {
                this->operator()(i, j) = lhl(i, j);
            }
        }

        return *this;
    }

    Matrix operator+(const Matrix& lhl)
    {
        if (m_rows != lhl.size().first or m_cols != lhl.size().second)
            throw std::runtime_error("Matrix::operator+() Matrixes are not compatible.");

        Matrix result(m_rows, m_cols);
        for (int i = 0; i < result.size().first; i++)
        {
            for (int j = 0; j < result.size().second; j++)
            {
                result(i, j) = this->operator()(i, j) + lhl(i, j);
            }
        }

        return result;
    }

    Matrix operator-(const Matrix& lhl)
    {
        if (m_rows != lhl.size().first or m_cols != lhl.size().second)
            throw std::runtime_error("Matrix::operator-() Matrixes are not compatible.");

        Matrix result(m_rows, m_cols);
        for (int i = 0; i < result.size().first; i++)
        {
            for (int j = 0; j < result.size().second; j++)
            {
                result(i, j) = this->operator()(i, j) - lhl(i, j);
            }
        }

        return result;
    }

    Matrix operator-(const std::vector<double>& lhl)
    {
        if (m_cols != 1 or lhl.size() != m_rows)
            throw std::runtime_error("Matrix::operator-() Matrixes are not compatible.");

        Matrix result(m_rows, m_cols);
        for (int i = 0; i < result.size().first; i++)
        {
            result(i, 0) = this->operator()(i, 0)- lhl.at(i);
        }

        return result;
    }

private:
    void _resize(unsigned int rows, unsigned int cols, double default_values = 0)
    {
        m_matrix.clear();

        m_rows = rows;
        m_cols = cols;

        m_matrix.resize(m_rows);
        for (int i = 0; i < m_rows; i++)
        {
            m_matrix[i].resize(m_cols, default_values);
        }
    }

private: 
    unsigned int m_rows;
    unsigned int m_cols;

    std::vector<std::vector<double>> m_matrix;
};