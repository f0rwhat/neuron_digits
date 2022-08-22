#pragma once

#include "IActivatorFunc.hpp"

#include <cmath>

class SigmoidFunc : public IActivatorFunc
{
public:
    double func(double x) override 
    {
        return 1 / (1 + exp(-x));
    }

    Matrix func(const Matrix& x) override
    {
        const auto size = x.size();
        Matrix result(size.first, size.second);

        for(int i = 0; i < size.first; i++)
        {
            for(int j = 0; j < size.second; j++)
            {
                result(i, j) = func(x(i, j));
            }
        }

        return result;
    }

    double derivative_func(double x) override
    {
        return exp(-x) / pow(1 + exp(-x), 2);
    }

    Matrix derivative_func(const Matrix& x) override
    {
        const auto size = x.size();
        Matrix result(size.first, size.second);

        for(int i = 0; i < size.first; i++)
        {
            for(int j = 0; j < size.second; j++)
            {
                result(i, j) = derivative_func(x(i, j));
            }
        }

        return result;
    }
};