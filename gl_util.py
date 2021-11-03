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

import OpenGL.GL as gl

class GlRgbTexture:
    def __init__(self, width, height):
        self.width = 0
        self.height = 0
        self.texture_id = 0
        self.have_data = False
        self.setup(width, height)
    
    def setup(self, width, height):
        self.width = width
        self.height = height

        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def update(self, data):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        if not self.have_data:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
                            self.width, self.height, 0, gl.GL_RGB,
                            gl.GL_UNSIGNED_BYTE, data)
            self.have_data = True
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D,
                               0, 0, 0,
                               self.width, self.height,
                               gl.GL_RGB,
                               gl.GL_UNSIGNED_BYTE,
                               data)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        
    def __del__(self):
        gl.glDeleteTextures(1, [self.texture_id])
