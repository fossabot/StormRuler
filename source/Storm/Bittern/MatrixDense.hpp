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

/// @brief Dense vector with fixed shape.
template<std::copyable Elem, size_t Extent>
  requires std::is_object_v<Elem> && (Extent > 0)
using FixedVector = FixedMatrix<Elem, Extent>;

/// @brief Fixed dense matrix (rank 1 specicalization).
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
    requires assignable_matrix<DenseMatrix, Matrix> &&
             (!std::same_as<DenseMatrix, std::remove_cvref_t<Matrix>>)
  constexpr explicit DenseMatrix(Matrix&& mat) noexcept
  {
    this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Assign the current matrix elements from matrix @p mat.
  template<matrix Matrix>
    requires assignable_matrix<DenseMatrix, Matrix> &&
             (!std::same_as<DenseMatrix, std::remove_cvref_t<Matrix>>)
  constexpr DenseMatrix& operator=(Matrix&& mat) noexcept
  {
    return this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Construct a matrix with elements @p elems.
  /// @{
  template<class... Elems>
    requires (Extent == sizeof...(Elems)) &&
             ((std::constructible_from<Elem, Elems> &&
               !std::same_as<DenseMatrix, std::remove_cvref_t<Elems>>) &&
              ...)
  constexpr explicit DenseMatrix(Elems&&... elems)
      : elems_{Elem{std::forward<Elems>(elems)}...}
  {
  }
  /// @}

  /// @brief Construct a matrix with element array @p elems.
  /// @{
  constexpr explicit DenseMatrix(array_rref_t<Elem, Extent> elems)
      : elems_{std::to_array(std::move(elems))}
  {
  }
  constexpr explicit DenseMatrix(array_cref_t<Elem, Extent> elems)
      : elems_{std::to_array(elems)}
  {
  }
  /// @}

  /// @brief Assign the current matrix elements from array @p elems.
  /// @{
  constexpr DenseMatrix& operator=(array_rref_t<Elem, Extent> elems) noexcept
  {
    elems_ = std::to_array(std::move(elems));
    return *this;
  }
  constexpr DenseMatrix& operator=(array_cref_t<Elem, Extent> elems) noexcept
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

/// @brief Fixed dense matrix (multirank specialization).
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
    requires assignable_matrix<DenseMatrix, Matrix> &&
             (!std::same_as<DenseMatrix, std::remove_cvref_t<Matrix>>)
  constexpr explicit DenseMatrix(Matrix&& mat) noexcept
  {
    this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Assign the current matrix elements from matrix @p mat.
  template<matrix Matrix>
    requires assignable_matrix<DenseMatrix, Matrix> &&
             (!std::same_as<DenseMatrix, std::remove_cvref_t<Matrix>>)
  constexpr DenseMatrix& operator=(Matrix&& mat) noexcept
  {
    return this->assign(std::forward<Matrix>(mat));
  }

  /// @brief Construct a matrix with slices @p slices.
  template<matrix... Slices>
    requires (Extent == sizeof...(Slices)) &&
             ((std::constructible_from<Slice_, Slices> &&
               !std::same_as<DenseMatrix, std::remove_cvref_t<Slices>>) &&
              ...)
  constexpr explicit DenseMatrix(Slices&&... slices)
      : slices_{Slice_{std::forward<Slices>(slices)}...}
  {
  }

  /// @brief Construct a matrix with slice array @p slices.
  /// @{
  constexpr explicit DenseMatrix(array_rref_t<Slice_, Extent> slices)
      : slices_{std::to_array(std::move(slices))}
  {
  }
  constexpr explicit DenseMatrix(array_cref_t<Slice_, Extent> slices)
      : slices_{std::to_array(slices)}
  {
  }
  /// @}

  /// @brief Assign the current matrix elements from array @p slices.
  /// @{
  constexpr DenseMatrix& operator=(array_rref_t<Slice_, Extent> slices) noexcept
  {
    slices_ = std::to_array(std::move(slices));
    return *this;
  }
  constexpr DenseMatrix& operator=(array_cref_t<Slice_, Extent> slices) noexcept
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
      array_cref_t<Subslices, SecondExtent_>... subslices)
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
DenseMatrix(array_cref_t<Elem, Extent>)
    -> DenseMatrix<std::remove_cvref_t<Elem>, fixed_shape_t<Extent>>;
template<matrix Slice, size_t Extent>
DenseMatrix(array_cref_t<Slice, Extent>)
    -> DenseMatrix<matrix_element_t<Slice>,
                   cat_shapes_t<fixed_shape_t<Extent>, matrix_shape_t<Slice>>>;

template<scalar... Elems, size_t SecondExtent>
  requires (sizeof...(Elems) > 1)
DenseMatrix(array_cref_t<Elems, SecondExtent>...)
    -> DenseMatrix<std::common_type_t<Elems...>, //
                   fixed_shape_t<sizeof...(Elems), SecondExtent>>;
template<matrix... Slices, size_t SecondExtent>
  requires (sizeof...(Slices) > 1)
DenseMatrix(array_cref_t<Slices, SecondExtent>...)
    -> DenseMatrix<std::common_type_t<matrix_element_t<Slices>...>,
                   cat_shapes_t<fixed_shape_t<sizeof...(Slices), SecondExtent>,
                                common_shape_t<matrix_shape_t<Slices>...>>>;

} // namespace Storm
