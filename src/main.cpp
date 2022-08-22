#include <GL/glut.h>
#include <thread>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <utility>
#include <cmath>

#include "RenderWindow.hpp"
#include "BitMap.hpp"
#include "Matrix.hpp"
#include "NeuroNet.hpp"
#include "activators/ModReluFunc.hpp"
#include "activators/SigmoidFunc.hpp"

const unsigned int WINDOW_WIDTH = 800;
const unsigned int WINDOW_HEIGHT = 560;
const unsigned int BLOCK_SIZE = 20;
const auto ROWS = 28; // WINDOW_HEIGHT / BLOCK_SIZE;
const auto COLUMNS = 28; // WINDOW_WIDTH / BLOCK_SIZE;

bool isMousePressed = false;

std::shared_ptr<RenderWindow> window;

void teach(std::shared_ptr<NeuroNet> neuroNet, unsigned int epoches, double percentToEnd = 0.97)
{
    std::ifstream input("lib_10k.txt");

    std::vector<std::pair<int, std::vector<double>>> teach_data;
    while(not input.eof())
    {
        int rightAnswer;
        std::vector<double> input_data;
        input_data.resize(784);

        input >> rightAnswer;
        for (int i = 0; i < 784; i++)
        {
            input >> input_data[i];
        }

        teach_data.push_back({rightAnswer, input_data});
    }
    std::cout << "Teaching data was read, starting learning process..." << std::endl;

    int good = 0;
    int total = 0;
    auto start_point = std::chrono::system_clock::now();
    double rate = 0;

    int learn_data_iterator = 0;
    double epoch = 0;
    while (epoch < epoches and rate < percentToEnd)
    {
        const auto& data = teach_data.at(learn_data_iterator);
        learn_data_iterator++;
        learn_data_iterator = learn_data_iterator < teach_data.size() ? learn_data_iterator : 0;


        const auto answer = neuroNet->analyze(data.second);

        if (answer == data.first)
        {
            good++;
        }
        else
        {
            neuroNet->back_propagate(data.first, 0.15 * exp(-epoch / static_cast<double>(epoches)));
        }
        total++;

        rate = good / static_cast<double>(total);

        if (total % 10000 == 0)
        {
            auto point_diff = std::chrono::system_clock::now() - start_point;
            auto time_spent_s = std::chrono::duration_cast<std::chrono::seconds>(point_diff);
            auto time_spent_m = std::chrono::duration_cast<std::chrono::minutes>(point_diff);
            std::cout << "Time spent: " << time_spent_m.count() << "m " << time_spent_s.count() % 60 << "s; Epoch: " << epoch  << "; Good: " << good << "; Total: " << total << "; Rate: " << rate << ";" << std::endl;
            epoch++;
            total = 0;
            good = 0;
        }
    }

    std::cout << "Teaching ended." << std::endl;
}

void main_loop(int)
{
    if (window)
        window->render();

    glutTimerFunc(1, main_loop, 0);
}

int main(int argc, char** argv)
{
    std::shared_ptr<IActivatorFunc> activator;
    std::shared_ptr<NeuroNet> neuroNet;
    int in = 0;

    std::cout << "Choose activate func:" << std::endl;
    std::cout << "1. ModRelu" << std::endl;
    std::cout << "2. Sigmoid" << std::endl;
    while (in != 1 and in != 2)
    {
        std::cin >> in;
        if (in != 1 and in != 2)
        {
            std::cout << "Incorrect input! Try again." << std::endl;
        }
    }
    switch (in)
    {
        case 1: 
            activator = std::make_shared<ModReluFunc>();
            break;
        case 2:
            activator = std::make_shared<SigmoidFunc>();
            break;
        default: 
            throw std::runtime_error("Unknown activator func.");
    }

    std::cout << "Read weights from \"weights.txt\"?" << std::endl;
    std::cout << "1. Yes" << std::endl;
    std::cout << "2. No" << std::endl;
    in = 0;
    while (in != 1 and in != 2)
    {
        std::cin >> in;
        if (in != 1 and in != 2)
        {
            std::cout << "Incorrect input! Try again." << std::endl;
        }
    }
    if (in == 1)
    {
        neuroNet = std::make_shared<NeuroNet>(std::vector<unsigned int>{784, 256, 10}, activator);
        neuroNet->read_weights("weights.txt");
    }
    else 
    {
        std::cout << "Input hidden layer size:" << std::endl;
        std::cin >> in;
        neuroNet = std::make_shared<NeuroNet>(std::vector<unsigned int>{784, static_cast<unsigned int>(in), 10}, activator);
    }

    std::cout << "Teach neuronet from  \"lib_10k.txt\"?" << std::endl;
    std::cout << "1. Yes" << std::endl;
    std::cout << "2. No" << std::endl;
    in = 0;
    while(true)
    {
        while (in != 1 and in != 2)
        {
            std::cin >> in;
            if (in != 1 and in != 2)
            {
                std::cout << "Incorrect input! Try again." << std::endl;
            }
        }
        if (in == 1)
        {
            std::cout << "Input teaching epoches count:" << std::endl;
            std::cin >> in;
            teach(neuroNet, in);
            neuroNet->save_weights("weights.txt");
        }
        std::cout << "Repeat?" << std::endl;
        std::cout << "1. Yes" << std::endl;
        std::cout << "2. No" << std::endl;
        in = 0;
        while (in != 1 and in != 2)
        {
            std::cin >> in;
            if (in != 1 and in != 2)
            {
                std::cout << "Incorrect input! Try again." << std::endl;
            }
        }
        if (in == 2)
            break;
    }

    std::shared_ptr<BitMap> bitMap = std::make_shared<BitMap>(ROWS, COLUMNS, BLOCK_SIZE);
    window = std::make_shared<RenderWindow>(WINDOW_WIDTH, WINDOW_HEIGHT, "Neuron");
    window->init();

    window->addObject(bitMap);

    static std::function<void(int, int, int, int)> mouseClick_bounce = [&] (int button, int state, int x, int y) {
        if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
            isMousePressed = state == GLUT_DOWN;
        }
    };
    auto mouseClick = [](int button, int state, int x, int y) {
        mouseClick_bounce(button, state, x, y);
    };

    const auto analyze = [&](){
        const auto data = bitMap->asVector();
        const auto result = neuroNet->analyze(data);
        
        std::cout << "Its " << result << std::endl;;
    };

    static std::function<void(int, int)> mouseMove_bounce = [&] (int x, int y) {
        if(isMousePressed) {
            // std::cout << "Mouse pressed : " << x << ":" << y << std::endl;

            if (x > 0 and y > 0 and x < WINDOW_WIDTH and y < WINDOW_HEIGHT)
            {
                const auto x_index = x / BLOCK_SIZE;
                const auto y_index = y / BLOCK_SIZE;

                bitMap->enable(y_index, x_index, true);
                analyze();
            }
        }
    };
    auto mouseMove = [](int x, int y) {
        mouseMove_bounce(x, y);
    };

    static std::function<void(unsigned char, int, int)> keyBoard_bounce = [&] (unsigned char key, int x, int y) {
        switch(key)
        {
            case ' ':
                bitMap->reset();
                break;
            case 13:
                analyze();
                break;
            default:
                break;
        }
    };
    auto keyboardPress = [](unsigned char key, int x, int y) {
        keyBoard_bounce(key, x, y);
    };

    glutMouseFunc(mouseClick);
    glutMotionFunc(mouseMove);
    glutKeyboardFunc(keyboardPress);

    glutTimerFunc(1, main_loop, 0);

    window->start();

    return 0;
}