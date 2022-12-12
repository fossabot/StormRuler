// Copyright (C) 2020-2023 Oleg Butakov
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR Allocator PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
// SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#pragma once

#include <Storm/Base.hpp>

#include <Storm/Bittern/Matrix.hpp>
#include <Storm/Bittern/MatrixAlgorithms.hpp>
#include <Storm/Bittern/MatrixDense.hpp>

#include <array>
#include <concepts>

namespace Storm
{

// -----------------------------------------------------------------------------

/// @brief Fixed-sized (small) matrix.
template<class Elem, size_t NumRows, size_t NumCols>
using Mat = FixedMatrix<Elem, NumRows, NumCols>;

/// @brief 2x2 matrix.
template<class Elem>
using Mat2x2 = Mat<Elem, 2, 2>;

/// @brief 3x3 matrix.
template<class Elem>
using Mat3x3 = Mat<Elem, 3, 3>;

/// @brief 4x4 matrix.
template<class Elem>
using Mat4x4 = Mat<Elem, 4, 4>;

/// @brief Fixed-sized (small) vector.
template<class Elem, size_t NumRows>
using Vec = FixedVector<Elem, NumRows>;

/// @brief 2D vector.
template<class Elem>
using Vec2D = Vec<Elem, 2>;

/// @brief 3D vector.
template<class Elem>
using Vec3D = Vec<Elem, 3>;

/// @brief 4D vector.
template<class Elem>
using Vec4D = Vec<Elem, 4>;

} // namespace Storm
