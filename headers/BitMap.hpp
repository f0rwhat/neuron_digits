#pragma once

#include <GL/glut.h>
#include <IRenderable.hpp>
#include <vector>

#include "Matrix.hpp"

class BitMap: public IRenderable
{
public:
    BitMap(unsigned int rows, unsigned int columns, unsigned int block_size)
        : m_rows(rows)
        , m_columns(columns)
        , m_block_size(block_size)
        , m_bitMap(rows, columns, 0)
    {
    }

    void enable(unsigned int cell_i, unsigned int cell_j, bool enable)
    {
        if (cell_i >= 0 and cell_i < m_rows and cell_j >= 0 and cell_j < m_columns)
        {
            m_bitMap(cell_i, cell_j) = enable;
            if (cell_j > 0 and m_bitMap(cell_i, cell_j - 1) == 0)
            {
                m_bitMap(cell_i, cell_j - 1) = 0.8;
            }
            if (cell_i > 0 and m_bitMap(cell_i - 1, cell_j) == 0)
            {
                m_bitMap(cell_i - 1, cell_j) = 0.8;
            }
            if (cell_j < m_columns - 1 and m_bitMap(cell_i, cell_j + 1) == 0)
            {
                m_bitMap(cell_i , cell_j + 1) = 0.8;
            }
            if (cell_i < m_rows - 1 and m_bitMap(cell_i + 1, cell_j) == 0)
            {
                m_bitMap(cell_i + 1, cell_j) = 0.8;
            }
        }
    }

    void reset()
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_columns; j++)
            {
                m_bitMap(i, j) = 0;
            }
        }
    }

    void render() override
    {
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_columns; j++)
            {
                const auto value = m_bitMap(i, j);
                draw_cell(i, j, value, value, value);
            }
        }
    }

    std::vector<double> asVector()
    {
        std::vector<double> result;
        for (int i = 0; i < m_rows; i++)
        {
            for (int j = 0; j < m_columns; j++)
            {
                result.push_back(m_bitMap(i, j));
            }
        }
        return result;
    }

private:
    void draw_cell(unsigned int i, unsigned int j, double r, double g, double b)
    {
        glColor3f(r, g, b);
        glBegin(GL_POLYGON);
        glVertex3f(j * m_block_size, i * m_block_size, 0.0);
        glVertex3f(j * m_block_size + m_block_size, i * m_block_size, 0.0);
        glVertex3f(j * m_block_size + m_block_size, i * m_block_size + m_block_size, 0.0);
        glVertex3f(j * m_block_size, i * m_block_size + m_block_size, 0.0);
        glEnd();
    }

private:
    const unsigned int m_rows;
    const unsigned int m_columns;
    const unsigned int m_block_size;
    Matrix m_bitMap;
};