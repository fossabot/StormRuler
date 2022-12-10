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

#include <array>
#include <concepts>

namespace Storm
{

// -----------------------------------------------------------------------------

/// @brief Dense matrix.
template<class Elem, class Shape>
class DenseMatrix;

template<matrix Matrix>
DenseMatrix(Matrix&&)
    -> DenseMatrix<matrix_element_t<Matrix>, matrix_shape_t<Matrix>>;

/// @brief Construct a dense matrix from @p args.
template<class... Args>
constexpr auto to_matrix(Args&&... args) noexcept
{
  return DenseMatrix{std::forward<Args>(args)...};
}

/// @brief Dense matrix type for arguments.
template<class... Args>
using matrix_t = decltype(to_matrix(std::declval<Args>()...));

// -----------------------------------------------------------------------------

/// @brief Dense matrix with fixed shape.
template<std::copyable Elem, size_t... Extents>
  requires std::is_object_v<Elem> && (... && (Extents > 0))
using FixedMatrix = DenseMatrix<Elem, fixed_shape_t<Extents...>>;

/// @brief Fixed matrix (rank 1).
template<std::copyable Elem, size_t Extent>
  requires std::is_object_v<Elem> && (Extent > 0)
class DenseMatrix<Elem, fixed_shape_t<Extent>> final :
    public TargetMatrixInterface<DenseMatrix<Elem, fixed_shape_t<Extent>>>
{
private:

  STORM_NO_UNIQUE_ADDRESS_ std::array<Elem, Extent> elems_{};

public:

  /// @brief Construct a matrix.
  constexpr DenseMatrix() = default;

  /// @brief Construct a matrix with elements of the matrix @p mat.
  template<matrix Matrix>
    requires matrix_assignable_from<DenseMatrix, Matrix>
  constexpr explicit DenseMatrix(Matrix&& mat) noexcept
  {
    this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Assign the current matrix elements from matrix @p mat.
  template<matrix Matrix>
    requires matrix_assignable_from<DenseMatrix, Matrix>
  constexpr DenseMatrix& operator=(Matrix&& mat) noexcept
  {
    return this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Construct a matrix with elements @p elems.
  /// @{
  template<class... Elems>
    requires (Extent == sizeof...(Elems)) &&
             ((std::constructible_from<Elem, Elems> &&
               !std::derived_from<std::remove_cvref_t<Elems>, DenseMatrix>) &&
              ...)
  constexpr explicit DenseMatrix(Elems&&... elems)
      : elems_{Elem{std::forward<Elems>(elems)}...}
  {
  }
  /// @}

  /// @brief Construct a matrix with element array @p elems.
  /// @{
  constexpr explicit DenseMatrix(std::array<Elem, Extent>&& elems)
      : elems_{std::move(elems)}
  {
  }
  constexpr explicit DenseMatrix(const std::array<Elem, Extent>& elems)
      : elems_{elems}
  {
  }
  constexpr explicit DenseMatrix(Elem (&&elems)[Extent])
      : elems_{std::to_array(std::move(elems))}
  {
  }
  constexpr explicit DenseMatrix(const Elem (&elems)[Extent])
      : elems_{std::to_array(elems)}
  {
  }
  /// @}

  /// @brief Assign the current matrix elements from array @p elems.
  /// @{
  constexpr DenseMatrix& operator=(Elem (&&elems)[Extent]) noexcept
  {
    elems_ = std::to_array(std::move(elems));
    return *this;
  }
  constexpr DenseMatrix& operator=(const Elem (&elems)[Extent]) noexcept
  {
    elems_ = std::to_array(elems);
    return *this;
  }
  /// @}

  /// @brief Get the matrix shape.
  constexpr auto shape() const noexcept
  {
    return shp<Extent>();
  }

  /// @brief Get the matrix element at @p index.
  /// @{
  constexpr Elem& operator()(size_t index) noexcept
  {
    STORM_ASSERT_(in_range(shape(), index), "Index is out of range!");
    return elems_[index];
  }
  constexpr const Elem& operator()(size_t index) const noexcept
  {
    STORM_ASSERT_(in_range(shape(), index), "Index is out of range!");
    return elems_[index];
  }
  /// @}

  /// @todo Transition code! Remove me!
  /// @{
  constexpr Elem& operator[](size_t row_index) noexcept
  {
    return (*this)(row_index);
  }
  constexpr const Elem& operator[](size_t row_index) const noexcept
  {
    return (*this)(row_index);
  }
  /// @}

}; // class DenseMatrix

/// @brief Fixed matrix (multirank).
template<std::copyable Elem, size_t Extent, size_t... RestExtents>
  requires std::is_object_v<Elem> && (Extent > 0) && (... && (RestExtents > 0))
class DenseMatrix<Elem, fixed_shape_t<Extent, RestExtents...>> final :
    public TargetMatrixInterface<
        DenseMatrix<Elem, fixed_shape_t<Extent, RestExtents...>>>
{
private:

  using Slice_ = FixedMatrix<Elem, RestExtents...>;
  static constexpr size_t SecondExtent_ = detail_::first_(RestExtents...);
  STORM_NO_UNIQUE_ADDRESS_ std::array<Slice_, Extent> slices_{};

public:

  /// @brief Construct a matrix.
  constexpr DenseMatrix() = default;

  /// @brief Construct a matrix with elements of the matrix @p mat.
  template<matrix Matrix>
    requires matrix_assignable_from<DenseMatrix, Matrix>
  constexpr explicit DenseMatrix(Matrix&& mat) noexcept
  {
    this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Assign the current matrix elements from matrix @p mat.
  template<matrix Matrix>
    requires matrix_assignable_from<DenseMatrix, Matrix>
  constexpr DenseMatrix& operator=(Matrix&& mat) noexcept
  {
    return this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Construct a matrix with slices @p slices.
  template<matrix... Slices>
    requires (Extent == sizeof...(Slices)) &&
             ((std::constructible_from<Slice_, Slices> &&
               !std::derived_from<std::remove_cvref_t<Slices>, DenseMatrix>) &&
              ...)
  constexpr explicit DenseMatrix(Slices&&... slices)
      : slices_{Slice_{std::forward<Slices>(slices)}...}
  {
  }

  /// @brief Construct a matrix with slice array @p slices.
  /// @{
  constexpr explicit DenseMatrix(std::array<Slice_, Extent>&& slices)
      : slices_{std::move(slices)}
  {
  }
  constexpr explicit DenseMatrix(const std::array<Slice_, Extent>& slices)
      : slices_{slices}
  {
  }
  constexpr explicit DenseMatrix(Slice_ (&&slices)[Extent])
      : slices_{std::to_array(std::move(slices))}
  {
  }
  constexpr explicit DenseMatrix(const Slice_ (&slices)[Extent])
      : slices_{std::to_array(slices)}
  {
  }
  /// @}

  /// @brief Assign the current matrix elements from array @p slices.
  /// @{
  constexpr DenseMatrix& operator=(Slice_ (&&slices)[Extent]) noexcept
  {
    slices_ = std::to_array(std::move(slices));
    return *this;
  }
  constexpr DenseMatrix& operator=(const Slice_ (&slices)[Extent]) noexcept
  {
    slices_ = std::to_array(slices);
    return *this;
  }
  /// @}

  template<class... Subslices>
    requires (Extent == sizeof...(Subslices)) &&
             (std::constructible_from<Slice_,
                                      matrix_t<Subslices[SecondExtent_]>> &&
              ...)
  constexpr explicit DenseMatrix(
      const Subslices (&... subslices)[SecondExtent_])
      : slices_{Slice_{to_matrix(subslices)}...}
  {
  }

  /// @brief Get the matrix shape.
  constexpr auto shape() const noexcept
  {
    return shp<Extent, RestExtents...>();
  }

  /// @brief Get the matrix element at @p indices.
  /// @{
  template<class... RestIndices>
    requires compatible_matrix_indices_v<DenseMatrix, size_t, RestIndices...>
  constexpr Elem& operator()(size_t index, //
                             RestIndices... rest_indices) noexcept
  {
    STORM_ASSERT_(in_range(shape(), index, rest_indices...),
                  "Indices are out of range!");
    return slices_[index](rest_indices...);
  }
  template<class... RestIndices>
    requires compatible_matrix_indices_v<DenseMatrix, size_t, RestIndices...>
  constexpr const Elem& operator()(size_t index,
                                   RestIndices... rest_indices) const noexcept
  {
    STORM_ASSERT_(in_range(shape(), index, rest_indices...),
                  "Indices are out of range!");
    return slices_[index](rest_indices...);
  }
  /// @}

}; // class DenseMatrix

template<scalar... Elems>
  requires (sizeof...(Elems) > 1)
DenseMatrix(Elems...) -> DenseMatrix<std::common_type_t<Elems...>,
                                     fixed_shape_t<sizeof...(Elems)>>;
template<matrix... Slices>
  requires (sizeof...(Slices) > 1)
DenseMatrix(Slices&&...)
    -> DenseMatrix<std::common_type_t<matrix_element_t<Slices>...>,
                   cat_shapes_t<fixed_shape_t<sizeof...(Slices)>,
                                common_shape_t<matrix_shape_t<Slices>...>>>;

template<scalar Elem, size_t Extent>
DenseMatrix(const Elem (&)[Extent])
    -> DenseMatrix<std::remove_cvref_t<Elem>, fixed_shape_t<Extent>>;
template<matrix Slice, size_t Extent>
DenseMatrix(const Slice (&)[Extent])
    -> DenseMatrix<matrix_element_t<Slice>,
                   cat_shapes_t<fixed_shape_t<Extent>, matrix_shape_t<Slice>>>;

template<scalar... Elems, size_t SecondExtent>
  requires (sizeof...(Elems) > 1)
DenseMatrix(const Elems (&... _)[SecondExtent]...)
    -> DenseMatrix<std::common_type_t<Elems...>, //
                   fixed_shape_t<sizeof...(Elems), SecondExtent>>;
template<matrix... Slices, size_t SecondExtent>
  requires (sizeof...(Slices) > 1)
DenseMatrix(const Slices (&... _)[SecondExtent])
    -> DenseMatrix<std::common_type_t<matrix_element_t<Slices>...>,
                   cat_shapes_t<fixed_shape_t<sizeof...(Slices), SecondExtent>,
                                common_shape_t<matrix_shape_t<Slices>...>>>;

// -----------------------------------------------------------------------------

/// @brief Statically-sized matrix.
/// @todo This is a legacy matrix. Do not use it!
template<class Elem, size_t NumRows, size_t NumCols>
class StaticMatrix final :
    public TargetMatrixInterface<StaticMatrix<Elem, NumRows, NumCols>>
{
private:

  std::array<Elem, NumRows * NumCols> elems_{};

public:

  /// @brief Construct a matrix.
  constexpr StaticMatrix(const Elem& init = {}) noexcept
  {
    fill(init);
  }

  /// @brief Construct a matrix with the elements.
  template<class... RestElems>
    requires (std::convertible_to<RestElems, Elem> && ...) &&
             (sizeof...(RestElems) + 1 == NumRows * NumCols)
  constexpr explicit StaticMatrix(const Elem& first_elem,
                                  const RestElems&... rest_elems) noexcept
      : elems_{first_elem, static_cast<Elem>(rest_elems)...}
  {
  }

  /// @brief Construct a matrix with another matrix.
  template<matrix Matrix>
  constexpr StaticMatrix(Matrix&& other) noexcept
  {
    assign_elements(*this, std::forward<Matrix>(other));
  }

  using TargetMatrixInterface<StaticMatrix<Elem, NumRows, NumCols>>::operator=;

  /// @brief Fill the matrix with @p value.
  constexpr void fill(const Elem& value) noexcept
  {
    for (size_t row_index = 0; row_index < NumRows; ++row_index) {
      for (size_t col_index = 0; col_index < NumCols; ++col_index) {
        (*this)(row_index, col_index) = value;
      }
    }
  }

  /// @brief Matrix shape.
  static constexpr auto shape() noexcept
  {
    return shp<NumRows, NumCols>();
  }

  /// @brief Get the matrix coefficient at @p row_index and @p col_index.
  /// @{
  constexpr Elem& operator()(size_t row_index, //
                             size_t col_index = 0) noexcept
  {
    STORM_ASSERT_(in_range(shape(), row_index, col_index),
                  "Indices are out of range!");
    return elems_[row_index * NumCols + col_index];
  }
  constexpr const Elem& operator()(size_t row_index,
                                   size_t col_index = 0) const noexcept
  {
    STORM_ASSERT_(in_range(shape(), row_index, col_index),
                  "Indices are out of range!");
    return elems_[row_index * NumCols + col_index];
  }
  /// @}

  /// @todo Transition code! Remove me!
  /// @{
  constexpr Elem& operator[](size_t row_index) noexcept
  {
    return (*this)(row_index);
  }
  constexpr const Elem& operator[](size_t row_index) const noexcept
  {
    return (*this)(row_index);
  }
  /// @}

}; // class StaticMatrix

/// @brief Statically-sized (small) matrix.
template<class Elem, size_t NumRows, size_t NumCols>
using Mat = StaticMatrix<Elem, NumRows, NumCols>;

/// @brief 2x2 matrix.
template<class Elem>
using Mat2x2 = Mat<Elem, 2, 2>;

/// @brief 3x3 matrix.
template<class Elem>
using Mat3x3 = Mat<Elem, 3, 3>;

/// @brief 4x4 matrix.
template<class Elem>
using Mat4x4 = Mat<Elem, 4, 4>;

/// @brief Statically-sized (small) vector.
template<class Elem, size_t NumRows>
using Vec = Mat<Elem, NumRows, 1>;

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
