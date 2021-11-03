# BSD 3-Clause License

# Copyright (c) 2021, Mark Liu
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from sdl2 import *
import OpenGL.GL as gl
from imgui.integrations.sdl2 import SDL2Renderer
import ctypes
import imgui
from gl_util import GlRgbTexture

class ImguiSdlWrapper:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self._rgb_textures = {}

        # Boilerplate adapted from https://github.com/pyimgui/pyimgui/blob/master/doc/examples/integrations_pysdl2.py
        # The original LICENSE file from the repo is reproduced below.

        # Copyright (c) 2016, Michał Jaworski
        # All rights reserved.

        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions are met:

        # * Redistributions of source code must retain the above copyright notice, this
        #   list of conditions and the following disclaimer.

        # * Redistributions in binary form must reproduce the above copyright notice,
        #   this list of conditions and the following disclaimer in the documentation
        #   and/or other materials provided with the distribution.

        # * Neither the name of pyimgui nor the names of its
        #   contributors may be used to endorse or promote products derived from
        #   this software without specific prior written permission.

        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
        # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
        # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
        # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
        # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        if SDL_Init(SDL_INIT_EVERYTHING) < 0:
            print("Error: SDL could not initialize! SDL Error: " + SDL_GetError().decode("utf-8"))
            exit(1)

        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24)
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1)
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1)
        SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1)
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE)
        SDL_SetHint(SDL_HINT_MAC_CTRL_CLICK_EMULATE_RIGHT_CLICK, b"1")
        SDL_SetHint(SDL_HINT_VIDEO_HIGHDPI_DISABLED, b"1")

        self.window = SDL_CreateWindow(self.name.encode('utf-8'),
                                  SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                  self.width, self.height,
                                  SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE)
        if self.window is None:
            print("Error: Window could not be created! SDL Error: " + SDL_GetError().decode("utf-8"))
            exit(1)

        self.gl_context = SDL_GL_CreateContext(self.window)

        if self.gl_context is None:
            print("Error: Cannot create OpenGL Context! SDL Error: " + SDL_GetError().decode("utf-8"))
            exit(1)

        SDL_GL_MakeCurrent(self.window, self.gl_context)

        if SDL_GL_SetSwapInterval(1) < 0:
            print("Warning: Unable to set VSync! SDL Error: " + SDL_GetError().decode("utf-8"))
            exit(1)

        imgui.create_context()
        self.impl = SDL2Renderer(self.window)

        self.event = SDL_Event()
        self.running = True

    def main_loop_begin(self):
        while SDL_PollEvent(ctypes.byref(self.event)) != 0:
            if self.event.type == SDL_QUIT:
                self.running = False
                break
            self.impl.process_event(self.event)
        self.impl.process_inputs()
        imgui.new_frame()                          

    def main_loop_end(self):
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        SDL_GL_SwapWindow(self.window)

    def add_image(self, image_id, image):
        # image should be numpy int8 array (height, width, 3)
        texture = GlRgbTexture(image.shape[1], image.shape[0])
        texture.update(image)
        self._rgb_textures[image_id] = texture
        return texture.texture_id

    def get_image(self, image_id):
        # texture_id usable for imgui.image(...)
        texture = self._rgb_textures[image_id]
        return (texture.texture_id, texture.width, texture.height)

    def update_image(self, image_id, image):
        # image should be numpy int8 array (height, width, 3)
        self._rgb_textures[image_id].update(image)

    def destroy(self):
        del self._rgb_textures
        self.impl.shutdown()
        SDL_GL_DeleteContext(self.gl_context)
        SDL_DestroyWindow(self.window)
