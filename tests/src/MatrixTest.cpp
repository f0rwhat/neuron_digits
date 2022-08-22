#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>

#include "Matrix.hpp"

TEST_CASE("Matrix::constructor and Matrix::destructor give no throw")
{
    const unsigned int rows = GENERATE(as<unsigned int>{}, 1, 256, 1000);
    const unsigned int cols = GENERATE(as<unsigned int>{}, 1, 256, 1000);
    const double default_values = GENERATE(as<double>{}, 0.5, 30);

    REQUIRE_NOTHROW([rows, cols, default_values](){
        Matrix* a = new Matrix(rows, cols, default_values);

        delete a;
    }());
}

TEST_CASE("Matrix::constructor creates matrix relevant to provided parameters")
{
    const unsigned int rows = GENERATE(as<unsigned int>{}, 1, 256, 1000);
    const unsigned int cols = GENERATE(as<unsigned int>{}, 1, 256, 1000);
    const double default_values = GENERATE(as<double>{}, 0.5, 30);

    Matrix a(rows, cols, default_values);
    REQUIRE(a.size().first == rows);
    REQUIRE(a.size().second == cols);

    REQUIRE_NOTHROW([&](){
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                REQUIRE(a(i, j) == default_values);
    }());
}

TEST_CASE("Matrix::transponate creates a correc transponent matrix")
{
    
    { //CASE _1
        Matrix a(3, 3, 0);
        a(0, 0) = 1; 
        a(0, 1) = 2; 
        a(0, 2) = 3; // 1 2 3
        a(1, 0) = 4; // 4 5 6
        a(1, 1) = 5; // 7 8 9
        a(1, 2) = 6;
        a(2, 0) = 7; 
        a(2, 1) = 8; 
        a(2, 2) = 9;

        auto b = Matrix::transponate(a);

        auto a_size = a.size();
        auto b_size = b.size();
        REQUIRE(b_size.first == a_size.second);
        REQUIRE(b_size.second == a_size.first);

        for (int i = 0; i < a_size.first; i++)
            for (int j = 0; j < a_size.second; j++)
                REQUIRE(a(i, j) == b(j, i));
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(2, 3, 0);
        a(0, 0) = 1; 
        a(0, 1) = 2; 
        a(0, 2) = 3; // 1 2 3
        a(1, 0) = 4; // 4 5 6
        a(1, 1) = 5;
        a(1, 2) = 6;

        auto b = Matrix::transponate(a);

        auto a_size = a.size();
        auto b_size = b.size();
        REQUIRE(b_size.first == a_size.second);
        REQUIRE(b_size.second == a_size.first);

        for (int i = 0; i < a_size.first; i++)
            for (int j = 0; j < a_size.second; j++)
                REQUIRE(a(i, j) == b(j, i));
    } // END_OF_CASE_2

    { // CASE_3
        Matrix a(3, 2, 0);
        a(0, 0) = 1; 
        a(0, 1) = 2; // 1 2 
        a(1, 0) = 3; // 3 4
        a(1, 1) = 4; // 5 6
        a(2, 0) = 5;
        a(2, 1) = 6;

        auto b = Matrix::transponate(a);

        auto a_size = a.size();
        auto b_size = b.size();
        REQUIRE(b_size.first == a_size.second);
        REQUIRE(b_size.second == a_size.first);

        for (int i = 0; i < a_size.first; i++)
            for (int j = 0; j < a_size.second; j++)
                REQUIRE(a(i, j) == b(j, i));
    } // END_OF_CASE_3
}

TEST_CASE("Matrix::operator() returns rvalue that allows to modify the value")
{
    const double default_values = GENERATE(as<double>{}, 0.5, 30);
    Matrix a(5, 5, default_values);
    
    REQUIRE(a(0, 0) == default_values);
    
    const double new_value = GENERATE(as<double>{}, 3.34, 21.7);
    a(0, 0) = new_value;

    REQUIRE(a(0, 0) == new_value);
}

TEST_CASE("Matrix::operator() throws an exception if out of bounds index was provided")
{
    const unsigned int rows = GENERATE(as<unsigned int>{}, 1, 1000);
    const unsigned int cols = GENERATE(as<unsigned int>{}, 1, 1000);
    Matrix a(rows, cols, 0);

    for (int t_case = 0; t_case < 2; t_case++)
    {
        unsigned int row;
        unsigned int col;
        switch (t_case)
        {
            case 0:
                row = rows;
                col = 0;
                break;
            case 1:
                row = 0;
                col = cols;
                break;
            default: break;
        }
        
        REQUIRE_THROWS_AS([&](){
            const auto temp = a(row, col);
        }(), std::exception);
    }
}

TEST_CASE("Matrix::operator*() performs correct multiplications of matrixes")
{
    { // CASE_1
        Matrix a(3, 3, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(0, 2) = 3;
        a(1, 0) = 4;
        a(1, 1) = 5;
        a(1, 2) = 6;
        a(2, 0) = 7;
        a(2, 1) = 8;
        a(2, 2) = 9;

        Matrix b(3, 2, 0);
        b(0, 0) = 1;
        b(0, 1) = 2;
        b(1, 0) = 3;
        b(1, 1) = 4;
        b(2, 0) = 5;
        b(2, 1) = 6;

        Matrix c = a * b;

        REQUIRE(c.size().first == a.size().first);
        REQUIRE(c.size().second == b.size().second);

        REQUIRE(c(0, 0) == 22);
        REQUIRE(c(0, 1) == 28);
        REQUIRE(c(1, 0) == 49);
        REQUIRE(c(1, 1) == 64);
        REQUIRE(c(2, 0) == 76);
        REQUIRE(c(2, 1) == 100);

        REQUIRE_THROWS_AS([&](){
            Matrix d = b * a;
        }(), std::exception);
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(3, 2, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        a(2, 0) = 5;
        a(2, 1) = 6;

        Matrix b(2, 1, 0);
        b(0, 0) = 1;
        b(1, 0) = 2;

        Matrix c = a * b;

        REQUIRE(c.size().first == a.size().first);
        REQUIRE(c.size().second == b.size().second);

        REQUIRE(c(0, 0) == 5);
        REQUIRE(c(1, 0) == 11);
        REQUIRE(c(2, 0) == 17);

        REQUIRE_THROWS_AS([&](){
            Matrix d = b * a;
        }(), std::exception);
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator*() performs correct multiplications of matrix and number")
{
    { // CASE_1
        Matrix a(3, 3, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(0, 2) = 3;
        a(1, 0) = 4;
        a(1, 1) = 5;
        a(1, 2) = 6;
        a(2, 0) = 7;
        a(2, 1) = 8;
        a(2, 2) = 9;

        Matrix b = a * 10;

        REQUIRE(b.size().first == a.size().first);
        REQUIRE(b.size().second == a.size().second);

        REQUIRE(b(0, 0) == 10);
        REQUIRE(b(0, 1) == 20);
        REQUIRE(b(0, 2) == 30);
        REQUIRE(b(1, 0) == 40);
        REQUIRE(b(1, 1) == 50);
        REQUIRE(b(1, 2) == 60);
        REQUIRE(b(2, 0) == 70);
        REQUIRE(b(2, 1) == 80);
        REQUIRE(b(2, 2) == 90);
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(3, 2, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        a(2, 0) = 5;
        a(2, 1) = 6;

        Matrix b = a * 0.01;

        REQUIRE(b.size().first == a.size().first);
        REQUIRE(b.size().second == a.size().second);

        REQUIRE(b(0, 0) == 0.01);
        REQUIRE(b(0, 1) == 0.02);
        REQUIRE(b(1, 0) == 0.03);
        REQUIRE(b(1, 1) == 0.04);
        REQUIRE(b(2, 0) == 0.05);
        REQUIRE(b(2, 1) == 0.06);
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator=() converts vector into single row matrix")
{
    { // CASE_1
        std::vector<double> test_vector = {1.5, 2.5, 3.1, 4.9, 5};
        Matrix a = test_vector;

        REQUIRE(a.size().first == test_vector.size());

        for (int i = 0; i < test_vector.size(); i++)
        {
            REQUIRE(a(i, 0) == test_vector.at(i));
        }
        
    } // END_OF_CASE_1

    { // CASE_2
        std::vector<double> test_vector = {1.5, 2.5, 3.1, 4.9, 5};
        Matrix a(100, 100);
        a = test_vector;

        REQUIRE(a.size().first == test_vector.size());

        for (int i = 0; i < test_vector.size(); i++)
        {
            REQUIRE(a(i, 0) == test_vector.at(i));
        }
        
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator=() equates rvl to lvl")
{
    { // CASE_1
        Matrix a(10, 6);
        for (int i = 0; i < a.size().first; i++)
        {
            for (int j = 0; j < a.size().second; j++)
            {
                a(i, j) = static_cast<double>(rand()) / RAND_MAX;
            }
        }

        Matrix b = a;

        REQUIRE(b.size().first == a.size().first);
        REQUIRE(b.size().second == a.size().second);

        for (int i = 0; i < a.size().first; i++)
        {
            for (int j = 0; j < a.size().second; j++)
            {
                REQUIRE(std::abs(b(i, j) - a(i, j)) < 0.01);
            }
        }
        
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(10, 6);
        for (int i = 0; i < a.size().first; i++)
        {
            for (int j = 0; j < a.size().second; j++)
            {
                a(i, j) = static_cast<double>(rand()) / RAND_MAX;
            }
        }

        Matrix b(115, 30, 0.5);

        b = a;

        REQUIRE(b.size().first == a.size().first);
        REQUIRE(b.size().second == a.size().second);

        for (int i = 0; i < a.size().first; i++)
        {
            for (int j = 0; j < a.size().second; j++)
            {
                REQUIRE(std::abs(b(i, j) - a(i, j)) < 0.01);
            }
        }
        
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator+() performs correct summation of matrixes")
{
    { // CASE_1
        Matrix a(3, 3, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(0, 2) = 3;
        a(1, 0) = 4;
        a(1, 1) = 5;
        a(1, 2) = 6;
        a(2, 0) = 7;
        a(2, 1) = 8;
        a(2, 2) = 9;

        Matrix b(3, 3, 0);
        b(0, 0) = 1;
        b(0, 1) = 2;
        b(0, 2) = 3;
        b(1, 0) = 4;
        b(1, 1) = 5;
        b(1, 2) = 6;
        b(2, 0) = 7;
        b(2, 1) = 8;
        b(2, 2) = 9;

        Matrix c = a + b;

        REQUIRE(c.size().first == a.size().first);
        REQUIRE(c.size().second == a.size().second);

        REQUIRE(c(0, 0) == 2);
        REQUIRE(c(0, 1) == 4);
        REQUIRE(c(0, 2) == 6);
        REQUIRE(c(1, 0) == 8);
        REQUIRE(c(1, 1) == 10);
        REQUIRE(c(1, 2) == 12);
        REQUIRE(c(2, 0) == 14);
        REQUIRE(c(2, 1) == 16);
        REQUIRE(c(2, 2) == 18);

        Matrix d = b + a;
        REQUIRE(d.size().first == a.size().first);
        REQUIRE(d.size().second == a.size().second);

        REQUIRE(d(0, 0) == 2);
        REQUIRE(d(0, 1) == 4);
        REQUIRE(d(0, 2) == 6);
        REQUIRE(d(1, 0) == 8);
        REQUIRE(d(1, 1) == 10);
        REQUIRE(d(1, 2) == 12);
        REQUIRE(d(2, 0) == 14);
        REQUIRE(d(2, 1) == 16);
        REQUIRE(d(2, 2) == 18);
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(3, 2, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        a(2, 0) = 5;
        a(2, 1) = 6;

        Matrix b(2, 1, 0);
        b(0, 0) = 1;
        b(1, 0) = 2;

        REQUIRE_THROWS_AS([&](){
            Matrix c = a + b;
        }(), std::exception);

        REQUIRE_THROWS_AS([&](){
            Matrix c = b + a;
        }(), std::exception);
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator-() performs correct substraction of matrixes")
{
    { // CASE_1
        Matrix a(3, 3, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(0, 2) = 3;
        a(1, 0) = 4;
        a(1, 1) = 5;
        a(1, 2) = 6;
        a(2, 0) = 7;
        a(2, 1) = 8;
        a(2, 2) = 9;

        Matrix b(3, 3, 0);
        b(0, 0) = 11;
        b(0, 1) = 12;
        b(0, 2) = 13;
        b(1, 0) = 14;
        b(1, 1) = 15;
        b(1, 2) = 16;
        b(2, 0) = 17;
        b(2, 1) = 18;
        b(2, 2) = 19;

        Matrix c = a - b;

        REQUIRE(c.size().first == a.size().first);
        REQUIRE(c.size().second == a.size().second);

        REQUIRE(c(0, 0) == -10);
        REQUIRE(c(0, 1) == -10);
        REQUIRE(c(0, 2) == -10);
        REQUIRE(c(1, 0) == -10);
        REQUIRE(c(1, 1) == -10);
        REQUIRE(c(1, 2) == -10);
        REQUIRE(c(2, 0) == -10);
        REQUIRE(c(2, 1) == -10);
        REQUIRE(c(2, 2) == -10);

        Matrix d = b - a;
        REQUIRE(d.size().first == a.size().first);
        REQUIRE(d.size().second == a.size().second);

        REQUIRE(d(0, 0) == 10);
        REQUIRE(d(0, 1) == 10);
        REQUIRE(d(0, 2) == 10);
        REQUIRE(d(1, 0) == 10);
        REQUIRE(d(1, 1) == 10);
        REQUIRE(d(1, 2) == 10);
        REQUIRE(d(2, 0) == 10);
        REQUIRE(d(2, 1) == 10);
        REQUIRE(d(2, 2) == 10);
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(3, 2, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        a(2, 0) = 5;
        a(2, 1) = 6;

        Matrix b(2, 1, 0);
        b(0, 0) = 1;
        b(1, 0) = 2;

        REQUIRE_THROWS_AS([&](){
            Matrix c = a - b;
        }(), std::exception);

        REQUIRE_THROWS_AS([&](){
            Matrix c = b - a;
        }(), std::exception);
    } // END_OF_CASE_2
}

TEST_CASE("Matrix::operator-() performs correct substraction of single row matrix and vector")
{
    { // CASE_1
        Matrix a(3, 1, 0);
        a(0, 0) = 1;
        a(1, 0) = 2;
        a(2, 0) = 3;

        std::vector<double> test_vector = {11, 12, 13};

        Matrix c = a - test_vector;

        REQUIRE(c.size().first == a.size().first);
        REQUIRE(c.size().second == 1);

        REQUIRE(c(0, 0) == -10);
        REQUIRE(c(1, 0) == -10);
        REQUIRE(c(2, 0) == -10);
    } // END_OF_CASE_1

    { // CASE_2
        Matrix a(3, 2, 0);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        a(2, 0) = 5;
        a(2, 1) = 6;

        std::vector<double> test_vector = {11, 12, 13};

        REQUIRE_THROWS_AS([&](){
            Matrix c = a - test_vector; 
        }(), std::exception);
    } // END_OF_CASE_2
}