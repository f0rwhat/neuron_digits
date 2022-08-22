#pragma once

#include "IActivatorFunc.hpp"

class ModReluFunc : public IActivatorFunc
{
public:
    double func(double x) override 
    {
        if (x < 0)
            return 0.01 * x;
        else if (0 <= x and x <= 1)
            return x;
        else // x > 1
            return 1 + 0.01 * (x - 1);
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
        if (x < 0)
            return 0.01;
        else if (0 <= x and x <= 1)
            return 1;
        else // x > 1
            return 0.01;
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