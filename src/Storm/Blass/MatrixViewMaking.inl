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
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
/// IN THE SOFTWARE.

#ifndef STORM_INSIDE_MATRIX_VIEW_HPP_
#error Do not include this header directly, \
       use <Storm/Blass/MatrixView.hpp> instead.
#endif

#include <concepts>
#include <type_traits>
#include <utility>

namespace Storm {

/// @brief Matrix generating view.
// clang-format off
template<std::copy_constructible Func>
  requires std::is_object_v<Func> &&
           std::regular_invocable<Func, size_t, size_t>
class MakeMatrixView final : public MatrixViewInterface<MakeMatrixView<Func>> {
  // clang-format on
private:

  matrix_shape_t shape_;
  STORM_NO_UNIQUE_ADDRESS_ Func func_;

public:

  /// @brief Construct a generating view.
  constexpr MakeMatrixView(matrix_shape_t shape, Func func)
      : shape_{shape}, func_{std::move(func)} {}

  /// @copydoc MatrixViewInterface::shape
  [[nodiscard]] constexpr matrix_shape_t shape() const noexcept {
    return shape_;
  }

  /// @copydoc MatrixViewInterface::operator()
  [[nodiscard]] constexpr decltype(auto)
  operator()(size_t row_index, size_t col_index) const noexcept {
    STORM_ASSERT_(row_index < shape_.num_rows && col_index < shape_.num_cols,
                  "Indices are out of range.");
    return func_(row_index, col_index);
  }

  /// @copydoc MatrixViewInterface::traverse_tree
  constexpr void traverse_tree(const auto& visitor) const {
    visitor(*this);
  }

  /// @copydoc MatrixViewInterface::transform_tree
  [[nodiscard]] constexpr auto transform_tree(const auto& transformer) const {
    return transformer(*this);
  }

}; // class MakeMatrixView

/// @brief Generate a constant matrix of @p shape.
/// @param value Matrix element value.
template<std::copyable Element>
[[nodiscard]] constexpr auto //
make_constant_matrix(matrix_shape_t shape, Element value) {
  return MakeMatrixView( //
      shape,
      [value = std::move(value)]( //
          [[maybe_unused]] size_t row_index,
          [[maybe_unused]] size_t col_index) noexcept -> const Element& {
        return value;
      });
}

/// @brief Generate a diagonal matrix of @p shape.
/// @param diagonal Matrix diagonal element value.
/// @param off_diagonal Matrix off-diagonal element value.
template<std::copyable Element>
[[nodiscard]] constexpr auto
make_diagonal_matrix(matrix_shape_t shape, //
                     Element diagonal, Element off_diagonal = Element{}) {
  return MakeMatrixView(
      shape,
      [diagonal = std::move(diagonal), //
       off_diagonal = std::move(off_diagonal)](
          size_t row_index, size_t col_index) noexcept -> const Element& {
        return (row_index == col_index) ? diagonal : off_diagonal;
      });
}

} // namespace Storm