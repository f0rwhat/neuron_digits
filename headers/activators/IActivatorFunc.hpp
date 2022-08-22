#pragma once

#include "Matrix.hpp"

class IActivatorFunc
{
public:
    virtual double func(double x) = 0; 
    virtual Matrix func(const Matrix& x) = 0; 
    virtual double derivative_func(double x) = 0;
    virtual Matrix derivative_func(const Matrix& x) = 0;
};