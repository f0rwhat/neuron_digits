#pragma once
#include <GL/glut.h>
#include <atomic>
#include <string>
#include <functional>
#include <list>
#include <memory>

#include "IRenderable.hpp"

class RenderWindow
{
public:
    RenderWindow(unsigned int width, unsigned int height, const std::string& title)
        : m_width(width)
        , m_height(height)
        , m_title(title)
    {}

    void init()
    {
        int my_argc = 0;

        // initialisation
        glutInit(&my_argc, NULL);
        glutInitDisplayMode(GLUT_SINGLE);
        glutInitWindowSize(m_width, m_height);
        glutInitWindowPosition(100, 100);
        glutCreateWindow(m_title.c_str());

        // window properties
        glMatrixMode (GL_PROJECTION);
        glLoadIdentity ();
        glOrtho(0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT), 0, 0, 1);
        glDisable(GL_DEPTH_TEST);
        glMatrixMode (GL_MODELVIEW);
        glLoadIdentity ();

        // resize event handler
        static std::function<void()> resize_bounce = [this] () {
            _resizeWindow();
        };
        auto resize = [](int, int) {
            resize_bounce();
        };
        glutReshapeFunc(resize);

        static std::function<void()> render_bounce = [this] () {
            render();
        };
        auto render_lambda = []() {
            render_bounce();
        };
        glutDisplayFunc(render_lambda);

        m_initComplete = true;
    }

    void start()
    {
        glutMainLoop();
    }

    void render()
    {
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        for (const auto& object : m_renderObjects)
        {
            if (const auto shared_object = object.lock())
            {
                shared_object->render();
            }
        }
        glFlush();
    }

    void addObject(std::weak_ptr<IRenderable> obj)
    {
        m_renderObjects.push_back(obj);
    }

private:
    void _resizeWindow()
    {
        if (glutGet(GLUT_WINDOW_WIDTH) != m_width or glutGet(GLUT_WINDOW_HEIGHT) != m_height)
        {
            glutReshapeWindow(m_width, m_height);
        }
    }

private:
    std::atomic_bool m_initComplete { false };
    const unsigned int m_width;
    const unsigned int m_height;
    const std::string m_title;

    std::list<std::weak_ptr<IRenderable>> m_renderObjects;
};