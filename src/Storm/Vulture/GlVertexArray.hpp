/// Copyright (C) 2022 Oleg Butakov
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to
/// deal in the Software without restriction, including without limitation the
/// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR Allocator PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
/// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
/// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
/// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
/// DEALINGS IN THE SOFTWARE.

#pragma once

#include <Storm/Base.hpp>

#include <Storm/Vulture/GlBuffer.hpp>

#include <GL/glew.h>
#include <glm/glm.hpp>

namespace Storm::Vulture::gl {

/// @brief OpenGL type enumeration.
template<class Type>
inline constexpr GLenum vertex_attrib_type_v = 0;

/// @brief Number of the components in type.
template<class Type>
inline constexpr GLint vertex_attrib_length_v = 1;

/// @brief Vertex attribute type.
// clang-format off
template<class Type>
concept vertex_attrib_type = (vertex_attrib_type_v<Type> != 0);
// clang-format on

// Scalar types.
template<>
inline constexpr GLenum vertex_attrib_type_v<GLbyte> = GL_BYTE;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLubyte> = GL_UNSIGNED_BYTE;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLshort> = GL_SHORT;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLushort> = GL_UNSIGNED_SHORT;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLint> = GL_INT;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLuint> = GL_UNSIGNED_INT;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLfloat> = GL_FLOAT;
template<>
inline constexpr GLenum vertex_attrib_type_v<GLdouble> = GL_DOUBLE;

// GLM vector type.
template<glm::length_t Length, vertex_attrib_type Type>
inline constexpr GLenum
    vertex_attrib_type_v<glm::vec<Length, Type, glm::packed>> =
        vertex_attrib_type_v<Type>;
template<glm::length_t Length, vertex_attrib_type Type>
inline constexpr GLint
    vertex_attrib_length_v<glm::vec<Length, Type, glm::packed>> =
        (static_cast<GLint>(Length * vertex_attrib_length_v<Type>));

// GLM matrix type.
template<glm::length_t Rows, glm::length_t Cols, vertex_attrib_type Type>
inline constexpr GLenum
    vertex_attrib_type_v<glm::mat<Rows, Cols, Type, glm::packed>> =
        vertex_attrib_type_v<Type>;
template<glm::length_t Rows, glm::length_t Cols, vertex_attrib_type Type>
inline constexpr GLint
    vertex_attrib_length_v<glm::mat<Rows, Cols, Type, glm::packed>> =
        (static_cast<GLint>(Rows * Cols * vertex_attrib_length_v<Type>));

/// @brief OpenGL draw mode.
enum class DrawMode : GLenum {
  points = GL_POINTS,
  lines = GL_LINES,
  triangles = GL_TRIANGLES,
}; // enum class DrawMode

/// @brief OpenGL vertex array.
class VertexArray final : detail_::noncopyable_ {
private:

  GLuint vertex_array_id_;

public:

  /// @brief Construct a vertex array.
  VertexArray() {
    glGenVertexArrays(1, &vertex_array_id_);
  }

  /// @brief Move-construct a vertex array.
  VertexArray(VertexArray&&) = default;
  /// @brief Move-assign the vertex array.
  VertexArray& operator=(VertexArray&&) = default;

  /// @brief Destruct the vertex array.
  ~VertexArray() {
    glDeleteVertexArrays(1, &vertex_array_id_);
  }

  /// @brief Cast to vertex array ID.
  [[nodiscard]] constexpr operator GLuint() const noexcept {
    return vertex_array_id_;
  }

  /// @brief Build the vertex array buffer.
  /// @{
  template<vertex_attrib_type... Types>
  void build(const Buffer<Types>&... vertex_buffers) {
    glBindVertexArray(vertex_array_id_);
    attach_vertex_attribs_(vertex_buffers...);
  }
  template<vertex_attrib_type... Types>
  void build_indexed(const Buffer<GLuint>& index_buffer,
                     const Buffer<Types>&... vertex_buffers) {
    glBindVertexArray(vertex_array_id_);
    index_buffer.bind(BufferTarget::element_array_buffer);
    attach_vertex_attribs_(vertex_buffers...);
  }
  /// @}

  /// @brief Draw the vertex array.
  /// @{
  void draw(DrawMode mode, GLsizei count) const {
    glBindVertexArray(vertex_array_id_);
    glDrawArrays(static_cast<GLenum>(mode), /*first*/ 0, count);
  }
  void draw_indexed(DrawMode mode, GLsizei count) const {
    glBindVertexArray(vertex_array_id_);
    glDrawElements(static_cast<GLenum>(mode), count, GL_UNSIGNED_INT,
                   /*indices*/ nullptr);
  }
  /// @}

private:

  template<vertex_attrib_type... Types>
  static void attach_vertex_attribs_(const Buffer<Types>&... vertex_buffers) {
    GLuint index = 0;
    (attach_single_vertex_attrib(index++, vertex_buffers), ...);
  }

  template<vertex_attrib_type Type>
  static void attach_single_vertex_attrib(GLuint index,
                                          const Buffer<Type>& vertex_buffer) {
    vertex_buffer.bind(BufferTarget::array_buffer);
    glEnableVertexAttribArray(index);
    glVertexAttribPointer(index, //
                          vertex_attrib_length_v<Type>,
                          vertex_attrib_type_v<Type>,
                          /*normalized*/ GL_FALSE,
                          /*stride*/ 0,
                          /*offset*/ nullptr);
  }

}; // class VertexArray

} // namespace Storm::Vulture::gl