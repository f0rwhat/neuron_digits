#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

#include "IActivatorFunc.hpp"
#include "Matrix.hpp"

template <typename ActivatorFunc>
class NeuroNet
{
public:
    NeuroNet(const std::vector<unsigned int>& layers_sizes, double default_weights)
        : m_activator(std::make_unique<ActivatorFunc>())
        , m_layers_sizes(layers_sizes)
    {
        _rebuild(default_weights);

        for (auto& layer : m_weights)
        {
            for (int i = 0; i < layer.size().first; i++)
            {
                for (int j = 0; j < layer.size().second; j++)
                {
                    layer(i, j) = ((rand() % 50)) * 0.06 / (layer.size().first + 15);
                }
            }
        }
    }

    double analyze(const std::vector<double>& input)
    {
        if (input.size() != m_layers_sizes.at(0))
            throw std::runtime_error("Input data size doesn't match the actual input layer size (" + std::to_string(input.size()) + " != " + std::to_string(m_layers_sizes.at(0)) + ").");

        m_neurons_layers[0] = input;

        for(int layer = 0; layer < m_layers_sizes.size() - 1; ++layer)
        {
            m_neurons_layers[layer + 1] = m_weights[layer] * m_neurons_layers[layer];
            m_neurons_layers[layer + 1] = m_neurons_layers[layer + 1] + m_bioses[layer];

            m_neurons_layers[layer + 1] = m_activator->func(m_neurons_layers[layer + 1]);
        }

        const auto outputLayerNum = m_layers_sizes.size() - 1;
        double max = -__DBL_MAX__;
        int max_answer = -1;
        for (int i = 0; i < m_neurons_layers.at(outputLayerNum).size().first; i++)
        {
            if (m_neurons_layers.at(outputLayerNum)(i, 0) > max)
            {
                max = m_neurons_layers.at(outputLayerNum)(i, 0);
                max_answer = i;
            } 
        }
        return max_answer;
    }

    void back_propagate(int reference, double study_coef = 1)
    {
        const int outputLayerNum = m_layers_sizes.size() - 1;
        for (int i = 0; i < m_layers_sizes.at(outputLayerNum); i++)
        {
            double d = i == reference ? 1 : 0;
            m_sigmas.at(outputLayerNum)(i, 0) = (d - m_neurons_layers.at(outputLayerNum)(i, 0)) * m_activator->derivative_func(m_neurons_layers.at(outputLayerNum)(i,0));
        }

        for (int layer = outputLayerNum - 1; layer > 0; layer--)
        {         
            auto transponated = Matrix::transponate(m_weights.at(layer));
            m_sigmas[layer] = transponated * m_sigmas[layer + 1];

            for(auto i = 0; i < m_layers_sizes.at(layer); i++)
            {
                m_sigmas[layer](i, 0) *= m_activator->derivative_func(m_neurons_layers.at(layer)(i, 0));
            }
        }

        for (int layer = 0; layer < outputLayerNum; layer++)
        {            
            for (int i = 0; i < m_weights.at(layer).size().first; i++)
            {
                for (int j = 0; j < m_weights.at(layer).size().second; j++)
                {
                    m_weights[layer](i, j) += m_neurons_layers.at(layer)(j, 0) * m_sigmas[layer + 1](i, 0) * study_coef;
                }
            }

            m_bioses[layer] = m_bioses[layer] + m_sigmas[layer + 1] * study_coef;
        }

        // check_for_nan();
    }
    
    void check_for_nan()
    {
        for(auto layer : m_neurons_layers)
        {
            auto size = layer.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    if (layer(i, j) != layer(i, j))
                    {
                        throw std::runtime_error("NAN!");
                    }
                }
            }
        }

        for(auto layer : m_weights)
        {
            auto size = layer.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    if (layer(i, j) != layer(i, j))
                    {
                        throw std::runtime_error("NAN!");
                    }
                }
            }
        }
    }

    void read_weights(const std::string& filename)
    {
        std::ifstream input(filename);
        if (not input)
        {
            throw std::runtime_error("Couldn't find file \"" + filename + "\".");
        }

        int layers_count;
        input >> layers_count;
        m_layers_sizes.clear();

        for (int i = 0; i < layers_count; i++)
        {
            double temp;
            input >> temp;
            m_layers_sizes.push_back(temp);
        }
        
        _rebuild();
        int count = 0;

        for (auto& layer : m_weights)
        {
            const auto size = layer.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    double temp = 0;
                    input >> temp;
                    layer(i, j) = temp;
                    count++;
                }
            }
        }

        for (auto& bios : m_bioses)
        {
            const auto size = bios.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    input >> bios(i, j);
                    count++;
                }
            }
        }
        
        std::cout << "Weights were read successfully. Total weights count: " << count << std::endl;
    }

    void save_weights(const std::string& filename)
    {
        std::ofstream output(filename);
        if (not output)
        {
            throw std::runtime_error("Couldn't open file \"" + filename + "\".");
        }

        output << m_layers_sizes.size() << " ";

        for (auto layer_size : m_layers_sizes)
        {
            output << layer_size << " ";
        }
        output << std::endl;

        for (auto& layer : m_weights)
        {
            const auto size = layer.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    output << layer(i, j) << " ";
                }
            }
        }

        output << std::endl;

        for (auto& bios : m_bioses)
        {
            const auto size = bios.size();
            for (int i = 0; i < size.first; i++)
            {
                for (int j = 0; j < size.second; j++)
                {
                    output << bios(i, j) << " ";
                }
            }
        }
        
        std::cout << "Weights were wroten successfully." << std::endl;
    }
 
private:

    void _rebuild(double default_weights = 0.5)
    {
        m_neurons_layers.clear();
        m_sigmas.clear();
        m_weights.clear();
        m_bioses.clear();

        for (int i = 0; i < m_layers_sizes.size(); i++)
        {
            m_neurons_layers.push_back(Matrix(m_layers_sizes.at(i), 1));
            m_sigmas.push_back(Matrix(m_layers_sizes.at(i), 1));

            if (i < m_layers_sizes.size() - 1)
            {
                // m_sum_layers.push_back(Matrix(m_layers_sizes.at(i + 1), 1));
                m_weights.push_back(Matrix(m_layers_sizes.at(i + 1), m_layers_sizes.at(i), default_weights));
                m_bioses.push_back(Matrix(m_layers_sizes.at(i + 1), 1, default_weights));
            }
        }
    }

private:
    std::unique_ptr<IActivatorFunc> m_activator;
    std::vector<unsigned int> m_layers_sizes;
    std::vector<Matrix> m_sigmas;
    std::vector<Matrix> m_neurons_layers;
    std::vector<Matrix> m_weights;
    std::vector<Matrix> m_bioses;
};