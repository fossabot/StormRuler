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

#include <Storm/Bittern/Math.hpp>
#include <Storm/Bittern/Matrix.hpp>

#include <concepts>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Storm
{

// -----------------------------------------------------------------------------

/// @brief Matrix transpose view.
template<matrix_view Matrix>
class TransposeMatrixView final :
    public MatrixViewInterface<TransposeMatrixView<Matrix>>
{
private:

  STORM_NO_UNIQUE_ADDRESS_ Matrix mat_;

public:

  /// @brief Construct a matrix transpose view.
  constexpr explicit TransposeMatrixView(Matrix mat) : mat_{std::move(mat)} {}

  /// @brief Get the matrix shape.
  constexpr auto shape() const noexcept
  {
    return std::array{num_cols(mat_), num_rows(mat_)};
  }

  /// @brief Get the matrix element at @p indices.
  /// @{
  constexpr decltype(auto) operator()(size_t row_index,
                                      size_t col_index) noexcept
  {
    STORM_ASSERT_(in_range(shape(), row_index, col_index),
                  "Indices are out of range!");
    return mat_(col_index, row_index);
  }
  constexpr decltype(auto) operator()(size_t row_index,
                                      size_t col_index) const noexcept
  {
    STORM_ASSERT_(in_range(shape(), row_index, col_index),
                  "Indices are out of range!");
    return mat_(col_index, row_index);
  }
  /// @}

}; // TransposeMatrixView

template<class Matrix>
TransposeMatrixView(Matrix&&) -> TransposeMatrixView<matrix_view_t<Matrix>>;

/// @brief Transpose the matrix @p mat.
template<viewable_matrix Matrix>
constexpr auto transpose(Matrix&& mat)
{
  return TransposeMatrixView(std::forward<Matrix>(mat));
}

// -----------------------------------------------------------------------------

} // namespace Storm
