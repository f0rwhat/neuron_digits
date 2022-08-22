#pragma once

class IRenderable
{
public:
    virtual void render() = 0;

    // virtual void subscribeToReRenderReadyness() = 0;
};