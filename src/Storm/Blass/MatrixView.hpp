/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// Copyright (C) 2022 Oleg Butakov
///
/// Permission is hereby granted, free of charge, to any person
/// obtaining a copy of this software and associated documentation
/// files (the "Software"), to deal in the Software without
/// restriction, including without limitation the rights  to use,
/// copy, modify, merge, publish, distribute, sublicense, and/or
/// sell copies of the Software, and to permit persons to whom the
/// Software is furnished to do so, subject to the following
/// conditions:
///
/// The above copyright notice and this permission notice shall be
/// included in all copies or substantial portions of the Software.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
/// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
/// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
/// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
/// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
/// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
/// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
/// OTHER DEALINGS IN THE SOFTWARE.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///

#pragma once

#include <concepts>
#include <ostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include <Storm/Base.hpp>

#include <Storm/Utils/Math.hpp>

#include <Storm/Blass/MatrixBase.hpp>

namespace Storm {

/// ----------------------------------------------------------------- ///
/// @brief Base class for all matrix views.
/// ----------------------------------------------------------------- ///
// clang-format off
template<class Derived>
  requires std::is_class_v<Derived> &&
           std::same_as<Derived, std::remove_cv_t<Derived>>
class BaseMatrixView {
private:
}; // class BaseMatrixView
// clang-format on

/// @brief Types, enabled to be a matrix view.
/// @{
template<class T>
struct enable_matrix_view : std::false_type {};
// clang-format off
template<class T>
  requires std::is_base_of_v<BaseMatrixView<T>, T>
struct enable_matrix_view<T> : std::true_type {};
// clang-format on
template<class T>
inline constexpr bool enable_matrix_view_v{enable_matrix_view<T>::value};
/// @}

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @brief Matrix view concept.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
template<class MatrixView>
concept matrix_view = matrix<MatrixView> && enable_matrix_view_v<MatrixView>;

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @brief Matrix that can be safely casted into a matrix view.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
template<class Matrix>
concept viewable_matrix = matrix<Matrix> &&
    matrix_view<std::remove_cvref_t<Matrix>> ||
    (!matrix_view<std::remove_cvref_t<Matrix>> &&
     std::is_lvalue_reference_v<Matrix>);

template<class MatrixView>
concept matrix_view_object_ = matrix_view<std::remove_cv_t<MatrixView>>;

/// @name Matrix views.
/// @{

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @name Forwarding views.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @{

/// ----------------------------------------------------------------- ///
/// @brief Matrix reference view.
/// ----------------------------------------------------------------- ///
template<matrix_object_ Matrix>
class MatrixRefView : public BaseMatrixView<MatrixRefView<Matrix>> {
private:

  Matrix* mat_;

public:

  /// @brief Construct a matrix reference view.
  constexpr MatrixRefView(Matrix& mat) noexcept : mat_{&mat} {}

  /// @copydoc BaseMatrixView::num_rows
  constexpr auto num_rows() const noexcept {
    return mat_->num_rows();
  }

  /// @copydoc BaseMatrixView::num_cols
  constexpr auto num_cols() const noexcept {
    return mat_->num_cols();
  }

  /// @copydoc BaseMatrixView::operator()
  /// @{
  constexpr auto operator()(size_t row_index, size_t col_index) noexcept
      -> decltype(auto) {
    return (*mat_)(row_index, col_index);
  }
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept
      -> decltype(auto) {
    return std::as_const(*mat_)(row_index, col_index);
  }
  /// @}

}; // class MatrixRefView

/// @brief Copy the matrix view @p mat.
template<class MatrixView>
constexpr auto
forward_as_matrix_view(const BaseMatrixView<MatrixView>& mat) noexcept {
  return static_cast<const MatrixView&>(mat);
}

/// @brief Wrap the matrix @p mat in view a matrix view.
/// @{
template<matrix_object_ Matrix>
requires(!matrix_view_object_<std::decay_t<Matrix>>) //
    constexpr auto forward_as_matrix_view(Matrix& mat) noexcept {
  return MatrixRefView<Matrix>{mat};
}
/// @}

template<viewable_matrix Matrix>
using forward_as_matrix_view_t =
    decltype(forward_as_matrix_view(std::declval<Matrix>()));

/// @} // Forwarding views.

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @name Generating views.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @{

/// ----------------------------------------------------------------- ///
/// @brief Matrix generating view.
/// ----------------------------------------------------------------- ///
// clang-format off
template<convertible_to_size_t_object NumRowsType,
         convertible_to_size_t_object NumColsType,
         std::regular_invocable<size_t, size_t> Func>
  requires std::is_object_v<Func>
class MatrixGeneratingView :
    public BaseMatrixView<
        MatrixGeneratingView<NumRowsType, NumColsType, Func>> {
private:

  [[no_unique_address]] NumRowsType num_rows_;
  [[no_unique_address]] NumColsType num_cols_;
  [[no_unique_address]] Func func_;

public:

  /// @brief Construct a generating view.
  constexpr MatrixGeneratingView( //
      NumRowsType num_rows, NumColsType num_cols, Func func) noexcept
      : num_rows_{num_rows}, num_cols_{num_cols}, func_{std::move(func)} {}

  /// @copydoc BaseMatrixView::num_rows
  constexpr auto num_rows() const noexcept {
    return num_rows_;
  }

  /// @copydoc BaseMatrixView::num_cols
  constexpr auto num_cols() const noexcept {
    return num_cols_;
  }

  /// @copydoc BaseMatrixView::operator()
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept {
    STORM_ASSERT_(row_index < num_rows_ && col_index < num_cols_ &&
                  "Indices are out of range.");
    return func_(row_index, col_index);
  }

}; // class MatrixGeneratingView
// clang-format on

template<class NumRowsType, class NumColsType, class Func>
MatrixGeneratingView(NumRowsType, NumColsType, Func)
    -> MatrixGeneratingView<NumRowsType, NumColsType, Func>;

/// @brief Generate a constant matrix with @p num_rows and @p num_cols.
/// @{
template<class Value>
constexpr auto make_constant_matrix( //
    std::convertible_to<size_t> auto num_rows,
    std::convertible_to<size_t> auto num_cols, Value scal) {
  return MatrixGeneratingView(
      num_rows, num_cols,
      [scal](size_t row_index, size_t col_index) { return scal; });
}
template<size_t NumRows, size_t NumCols, class Value>
constexpr auto make_constant_matrix(Value scal) {
  return make_constant_matrix<Value>(size_t_constant<NumRows>{}, //
                                     size_t_constant<NumCols>{}, scal);
}
/// @}

/// @brief Generate a diagonal matrix with @p num_rows and @p num_cols.
/// @{
template<class Value, class Tag = void>
constexpr auto make_diagonal_matrix( //
    std::convertible_to<size_t> auto num_rows,
    std::convertible_to<size_t> auto num_cols, Value scal) {
  return MatrixGeneratingView(num_rows, num_cols,
                              [scal](size_t row_index, size_t col_index) {
                                constexpr Value zero{};
                                return row_index == col_index ? scal : zero;
                              });
}
template<size_t NumRows, size_t NumCols, class Value>
constexpr auto make_diagonal_matrix(Value scal) {
  return make_diagonal_matrix<Value>(size_t_constant<NumRows>{}, //
                                     size_t_constant<NumCols>{}, scal);
}
/// @} // Generating views.

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @name Slicing views.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @{

/// @brief All indices range.
class AllIndices {
public:

  constexpr auto size() const = delete;

  constexpr size_t operator[](size_t index) const noexcept {
    return index;
  }

}; // AllIndices

/// @brief Selected indices range.
template<size_t Size>
class SelectedIndices {
private:

  [[no_unique_address]] std::array<size_t, Size> selected_indices_;

public:

  constexpr explicit SelectedIndices(
      const std::array<size_t, Size>& selected_indices) noexcept
      : selected_indices_(selected_indices) {}

  constexpr auto size() const noexcept {
    return size_t_constant<Size>{};
  }

  constexpr size_t operator[](size_t index) const noexcept {
    STORM_ASSERT_(index < Size && "Index is out of range.");
    return selected_indices_[index];
  }

}; // class SelectedIndices

template<size_t Size>
SelectedIndices(std::array<size_t, Size>) -> SelectedIndices<Size>;

/// @brief Sliced indices range.
template<convertible_to_size_t_object FromType,
         convertible_to_size_t_object ToType,
         convertible_to_size_t_object StrideType>
class SlicedIndices {
private:

  [[no_unique_address]] FromType from_;
  [[no_unique_address]] ToType to_;
  [[no_unique_address]] StrideType stride_;

  constexpr static auto compute_size_(auto from, auto to, auto stride) {
    return (from - to) / stride;
  }
  template<class FromType_, class ToType_, class StrideType_, //
           FromType_ From, ToType_ To, StrideType_ Stride>
  constexpr static auto
  compute_size_(std::integral_constant<FromType_, From>,
                std::integral_constant<ToType_, To>,
                std::integral_constant<StrideType_, Stride>) {
    return size_t_constant<(From - To) / Stride>{};
  }

public:

  constexpr SlicedIndices( //
      FromType from, ToType to, StrideType stride) noexcept
      : from_{from}, to_{to}, stride_{stride} {}

  constexpr auto size() const noexcept {
    return compute_size_(from_, to_, stride_);
  }

  constexpr size_t operator[](size_t index) const noexcept {
    STORM_ASSERT_(index < size() && "Index is out of range.");
    return from_ + stride_ * index;
  }

}; // class SlicedIndices

template<class FromType, class ToType, class StrideType>
SlicedIndices(FromType, ToType, StrideType)
    -> SlicedIndices<FromType, ToType, StrideType>;

/// ----------------------------------------------------------------- ///
/// @brief Submatrix view.
/// ----------------------------------------------------------------- ///
// clang-format off
template<matrix Matrix, class RowIndices, class ColIndices>
  requires std::is_object_v<RowIndices> && std::is_object_v<ColIndices>
class SubmatrixView :
    public BaseMatrixView<SubmatrixView<Matrix, RowIndices, ColIndices>> {
private:

  [[no_unique_address]] Matrix mat_;
  [[no_unique_address]] RowIndices row_indices_;
  [[no_unique_address]] ColIndices col_indices_;

public:

  /// @brief Construct a matrix rows view.
  constexpr SubmatrixView(Matrix mat, //
                          RowIndices row_indices,
                          ColIndices col_indices) noexcept
      : mat_{std::move(mat)},                 //
        row_indices_{std::move(row_indices)}, //
        col_indices_{std::move(col_indices)} {}

  /// @copydoc BaseMatrixView::num_rows
  /// @{
  constexpr auto num_rows() const noexcept 
      requires requires { std::declval<RowIndices>().size(); } { 
    return row_indices_.size(); 
  }
  constexpr auto num_rows() const noexcept {
    return mat_.num_rows();
  }
  /// @}

  /// @copydoc BaseMatrixView::num_cols
  /// @{
  constexpr auto num_cols() const noexcept 
      requires requires { std::declval<ColIndices>().size(); } { 
    return col_indices_.size(); 
  }
  constexpr auto num_cols() const noexcept {
    return mat_.num_cols();
  }
  /// @}

  /// @copydoc BaseMatrixView::operator()
  /// @{
  constexpr auto operator()(size_t row_index, size_t col_index) noexcept
      -> decltype(auto) {
    return mat_(row_indices_[row_index], col_indices_[col_index]);
  }
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept
      -> decltype(auto) {
    return mat_(row_indices_[row_index], col_indices_[col_index]);
  }
  /// @}

}; // class SubmatrixView
// clang-format on

template<class Matrix, class RowIndices, class ColIndices>
SubmatrixView(Matrix&&, RowIndices, ColIndices)
    -> SubmatrixView<forward_as_matrix_view_t<Matrix>, RowIndices, ColIndices>;

/// @brief Select the matrix @p mat rows with @p row_indices view.
constexpr auto select_rows(viewable_matrix auto&& mat,
                           std::integral auto... row_indices) noexcept {
  STORM_ASSERT_((static_cast<size_t>(row_indices) < mat.num_rows()) && ... &&
                "Row indices are out of range.");
  return SubmatrixView(
      std::forward<decltype(mat)>(mat),
      SelectedIndices(std::array{static_cast<size_t>(row_indices)...}),
      AllIndices{});
}

/// @brief Select the matrix @p mat columns with @p col_index view.
constexpr auto select_cols(viewable_matrix auto&& mat,
                           std::integral auto... col_indices) noexcept {
  STORM_ASSERT_((static_cast<size_t>(col_indices) < mat.num_cols()) && ... &&
                "Columns indices are out of range.");
  return SubmatrixView(
      std::forward<decltype(mat)>(mat), //
      AllIndices{},
      SelectedIndices(std::array{static_cast<size_t>(col_indices)...}));
}

/// @brief Slice the matrix @p mat rows from index @p rows_from
///   to index @p rows_to (not including) with a stride @p row_stride view.
/// @{
constexpr auto slice_rows(viewable_matrix auto&& mat,
                          std::convertible_to<size_t> auto rows_from,
                          std::convertible_to<size_t> auto rows_to,
                          std::convertible_to<size_t> auto row_stride =
                              size_t_constant<1>{}) noexcept {
  STORM_ASSERT_(rows_from < rows_to && rows_to <= mat.num_rows() &&
                "Invalid row range.");
  return SubmatrixView(std::forward<decltype(mat)>(mat),
                       SlicedIndices(rows_from, rows_to, row_stride),
                       AllIndices{});
}
template<size_t RowsFrom, size_t RowsTo, size_t RowStride = 1>
constexpr auto slice_rows(viewable_matrix auto&& mat) {
  return slice_rows(std::forward<decltype(mat)>(mat),
                    size_t_constant<RowsFrom>{}, size_t_constant<RowsTo>{},
                    size_t_constant<RowStride>{});
}
/// @}

/// @brief Slice the matrix @p mat columns from index @p cols_from
///   to index @p cols_to (not including) with a stride @p col_stride view.
/// @{
constexpr auto slice_cols(viewable_matrix auto&& mat,
                          std::convertible_to<size_t> auto cols_from,
                          std::convertible_to<size_t> auto cols_to,
                          std::convertible_to<size_t> auto col_stride =
                              size_t_constant<1>{}) noexcept {
  STORM_ASSERT_(cols_from < cols_to && cols_to <= mat.num_cols() &&
                "Invalid column range.");
  return SubmatrixView(std::forward<decltype(mat)>(mat), //
                       AllIndices{},
                       SlicedIndices(cols_from, cols_to, col_stride));
}
template<size_t ColsFrom, size_t ColsTo, size_t ColStride = 1>
constexpr auto slice_cols(viewable_matrix auto&& mat) {
  return slice_cols(std::forward<decltype(mat)>(mat),
                    size_t_constant<ColsFrom>{}, size_t_constant<ColsTo>{},
                    size_t_constant<ColStride>{});
}
/// @}

/// @todo Implement me!
constexpr auto diag(viewable_matrix auto&& mat) noexcept;

/// @todo Implement me!
constexpr auto lower_triangle(viewable_matrix auto&& mat) noexcept;

/// @todo Implement me!
constexpr auto upper_triangle(viewable_matrix auto&& mat) noexcept;

/// @} // Slicing views.

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @name Functional views.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @{

/// ----------------------------------------------------------------- ///
/// @brief Component-wise product of function to matrices view.
/// ----------------------------------------------------------------- ///
// clang-format off
template<std::copy_constructible Func, matrix... Matrices>
  requires std::is_object_v<Func> &&
           std::regular_invocable<Func, matrix_reference_t<Matrices>...>
class MapMatrixView :
    public BaseMatrixView<MapMatrixView<Func, Matrices...>> {
private:

  [[no_unique_address]] Func func_;
  [[no_unique_address]] std::tuple<Matrices...> mats_;

public:

  /// @brief Construct a map view.
  constexpr MapMatrixView(Func func, Matrices... mats) noexcept
      : func_{std::move(func)}, mats_{std::move(mats)...} {
  }

  /// @copydoc BaseMatrixView::num_rows
  constexpr auto num_rows() const noexcept {
    return std::get<0>(mats_).num_rows();
  }

  /// @copydoc BaseMatrixView::num_cols
  constexpr auto num_cols() const noexcept {
    return std::get<0>(mats_).num_cols();
  }

  /// @copydoc BaseMatrixView::operator()
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept
      -> decltype(auto) {
    return std::apply(
        [&, row_index, col_index](const auto&... mats) -> decltype(auto) {
          return func_(mats(row_index, col_index)...);
        },
        mats_);
  }

}; // class MapMatrixView
// clang-format on

template<class Func, class... Matrices>
MapMatrixView(Func, Matrices&&...)
    -> MapMatrixView<Func, forward_as_matrix_view_t<Matrices>...>;

/// @brief Make a component-wise product of function @p func
///   to matrices @p mat1, @p mats view.
constexpr auto map(auto func, //
                   viewable_matrix auto&& mat1,
                   viewable_matrix auto&&... mats) noexcept {
  STORM_ASSERT_(((mat1.num_rows() == mats.num_rows()) && ...) &&
                ((mat1.num_cols() == mats.num_cols()) && ...) &&
                "Shapes of the matrix arguments should be the same.");
  return MapMatrixView(func, //
                       std::forward<decltype(mat1)>(mat1),
                       std::forward<decltype(mats)>(mats)...);
}

/// @brief "+" the matrix @p mat.
constexpr auto operator+(viewable_matrix auto&& mat) noexcept {
  return map([](const auto& val) { return +val; }, mat);
}

/// @brief Negate the matrix @p mat.
constexpr auto operator-(viewable_matrix auto&& mat) noexcept {
  return map([](const auto& val) { return -val; }, mat);
}

/// @brief Multiply the matrix @p mat by a scalar @p scal.
/// @{
/// @todo `scal` should really be auto!
constexpr auto operator*(real_t scal, viewable_matrix auto&& mat) noexcept {
  return map([scal](const auto& val) { return scal * val; }, mat);
}
constexpr auto operator*(viewable_matrix auto&& mat, real_t scal) noexcept {
  return map([scal](const auto& val) { return val * scal; }, mat);
}
/// @}

/// @brief Divide the matrix @p mat by a scalar @p scal.
/// @todo `scal` should really be auto!
constexpr auto operator/(viewable_matrix auto&& mat, real_t scal) noexcept {
  return map([scal](const auto& val) { return val / scal; }, mat);
}

/// @brief Add the matrices @p mat1 and @p mat2.
constexpr auto operator+(viewable_matrix auto&& mat1,
                         viewable_matrix auto&& mat2) noexcept {
  return map([](const auto& val1, const auto& val2) { return val1 + val2; },
             mat1, mat2);
}

/// @brief Subtract the matrices @p mat1 and @p mat2.
constexpr auto operator-(viewable_matrix auto&& mat1,
                         viewable_matrix auto&& mat2) noexcept {
  return map([](const auto& val1, const auto& val2) { return val1 - val2; },
             mat1, mat2);
}

/// @brief Component-wise multiply the matrices @p mat1 and @p mat2.
constexpr auto operator*(viewable_matrix auto&& mat1,
                         viewable_matrix auto&& mat2) noexcept {
  return map([](const auto& val1, const auto& val2) { return val1 * val2; },
             mat1, mat2);
}

/// @brief Component-wise divide the matrices @p mat1 and @p mat2.
constexpr auto operator/(viewable_matrix auto&& mat1,
                         viewable_matrix auto&& mat2) noexcept {
  return map([](const auto& val1, const auto& val2) { return val1 / val2; },
             mat1, mat2);
}

namespace math {

  /// @brief Component-wise @c abs of the matrix @p mat.
  constexpr auto abs(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::abs(val); }, mat);
  }

  /// @name Power functions.
  /// @{

  constexpr auto pow(viewable_matrix auto&& x_mat, auto y) noexcept {
    return map([y](const auto& x) { return math::pow(x, y); }, x_mat);
  }

  constexpr auto pow(auto x, viewable_matrix auto&& y_mat) noexcept {
    return map([x](const auto& y) { return math::pow(x, y); }, y_mat);
  }

  constexpr auto pow(viewable_matrix auto&& x_mat,
                     viewable_matrix auto&& y_mat) noexcept {
    return map([](const auto& x, const auto& y) { return math::pow(x, y); },
               x_mat, y_mat);
  }

  /// @brief Component-wise @c sqrt of the matrix @p mat.
  constexpr auto sqrt(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::sqrt(val); }, mat);
  }

  /// @brief Component-wise @c cbrt of the matrix @p mat.
  constexpr auto cbrt(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::cbrt(val); }, mat);
  }

  constexpr auto hypot(viewable_matrix auto&& x_mat,
                       viewable_matrix auto&& y_mat) noexcept {
    return map([](const auto& x, const auto& y) { return math::hypot(x, y); },
               x_mat, y_mat);
  }

  constexpr auto hypot(viewable_matrix auto&& x_mat,
                       viewable_matrix auto&& y_mat,
                       viewable_matrix auto&& z_mat) noexcept {
    return map([](const auto& x, const auto& y, const auto& z) //
               { return math::hypot(x, y, z); },
               x_mat, y_mat, z_mat);
  }

  /// @} // Power functions.

  /// @name Exponential functions.
  /// @{

  /// @brief Component-wise @c exp of the matrix @p mat.
  constexpr auto exp(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::exp(val); }, mat);
  }

  /// @brief Component-wise @c exp2 of the matrix @p mat.
  constexpr auto exp2(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::exp2(val); }, mat);
  }

  /// @brief Component-wise @c log of the matrix @p mat.
  constexpr auto log(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::log(val); }, mat);
  }

  /// @brief Component-wise @c log2 of the matrix @p mat.
  constexpr auto log2(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::log2(val); }, mat);
  }

  /// @brief Component-wise @c log10 of the matrix @p mat.
  constexpr auto log10(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::log10(val); }, mat);
  }

  /// @} // Exponential functions.

  /// @name Trigonometric functions.
  /// @{

  /// @brief Component-wise @c sin of the matrix @p mat.
  constexpr auto sin(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::sin(val); }, mat);
  }

  /// @brief Component-wise @c cos of the matrix @p mat.
  constexpr auto cos(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::cos(val); }, mat);
  }

  /// @brief Component-wise @c tan of the matrix @p mat.
  constexpr auto tan(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::tan(val); }, mat);
  }

  /// @brief Component-wise @c asin of the matrix @p mat.
  constexpr auto asin(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::asin(val); }, mat);
  }

  /// @brief Component-wise @c acos of the matrix @p mat.
  constexpr auto acos(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::acos(val); }, mat);
  }

  /// @brief Component-wise @c atan of the matrix @p mat.
  constexpr auto atan(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::atan(val); }, mat);
  }

  /// @brief Component-wise @c atan2 of the matriсes @p y_mat and @p x_mat.
  constexpr auto atan2(viewable_matrix auto&& y_mat,
                       viewable_matrix auto&& x_mat) noexcept {
    return map([](const auto& y, const auto& x) { return math::atan2(y, x); },
               y_mat, x_mat);
  }

  /// @} // Trigonometric functions.

  /// @name Hyperbolic functions.
  /// @{

  /// @brief Component-wise @p sinh of the matrix @p mat.
  constexpr auto sinh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::sinh(val); }, mat);
  }

  /// @brief Component-wise @c cosh of the matrix @p mat.
  constexpr auto cosh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::cosh(val); }, mat);
  }

  /// @brief Component-wise @c tanh of the matrix @p mat.
  constexpr auto tanh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::tanh(val); }, mat);
  }

  /// @brief Component-wise @c asinh of the matrix @p mat.
  constexpr auto asinh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::asinh(val); }, mat);
  }

  /// @brief Component-wise @c acosh of the matrix @p mat.
  constexpr auto acosh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::acosh(val); }, mat);
  }

  /// @brief Component-wise @c atanh of the matrix @p mat.
  constexpr auto atanh(viewable_matrix auto&& mat) noexcept {
    return map([](const auto& val) { return math::atanh(val); }, mat);
  }

  /// @} // Hyperbolic functions.

} // namespace math

/// ----------------------------------------------------------------- ///
/// @brief Matrix transpose view.
/// ----------------------------------------------------------------- ///
template<matrix Matrix>
class MatrixTransposeView : public BaseMatrixView<MatrixTransposeView<Matrix>> {
private:

  [[no_unique_address]] Matrix mat_;

public:

  /// @brief Construct a matrix transpose view.
  constexpr explicit MatrixTransposeView(Matrix mat) noexcept
      : mat_{std::move(mat)} {}

  /// @copydoc BaseMatrixView::num_rows
  constexpr auto num_rows() const noexcept {
    return mat_.num_cols();
  }

  /// @copydoc BaseMatrixView::num_cols
  constexpr auto num_cols() const noexcept {
    return mat_.num_rows();
  }

  /// @copydoc BaseMatrixView::operator()
  /// @{
  constexpr auto operator()(size_t row_index, size_t col_index) noexcept
      -> decltype(auto) {
    return mat_(col_index, row_index);
  }
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept
      -> decltype(auto) {
    return mat_(col_index, row_index);
  }
  /// @}

}; // MatrixTransposeView

template<class Matrix>
MatrixTransposeView(Matrix&&)
    -> MatrixTransposeView<forward_as_matrix_view_t<Matrix>>;

/// @brief Transpose the matrix @p mat.
constexpr auto transpose(viewable_matrix auto&& mat) noexcept {
  return MatrixTransposeView(std::forward<decltype(mat)>(mat));
}

/// ----------------------------------------------------------------- ///
/// @brief Matrix product view.
/// ----------------------------------------------------------------- ///
template<matrix Matrix1, matrix Matrix2>
class MatrixProductView :
    public BaseMatrixView<MatrixProductView<Matrix1, Matrix2>> {
private:

  [[no_unique_address]] Matrix1 mat1_;
  [[no_unique_address]] Matrix2 mat2_;

public:

  /// @brief Construct a matrix product view.
  constexpr explicit MatrixProductView(Matrix1 mat1, Matrix2 mat2) noexcept
      : mat1_{std::move(mat1)}, mat2_{std::move(mat2)} {}

  /// @copydoc BaseMatrixView::num_rows
  constexpr auto num_rows() const noexcept {
    return mat1_.num_rows();
  }

  /// @copydoc BaseMatrixView::num_cols
  constexpr auto num_cols() const noexcept {
    return mat2_.num_cols();
  }

  /// @copydoc BaseMatrixView::operator()
  constexpr auto operator()(size_t row_index, size_t col_index) const noexcept
      -> decltype(auto) {
    const auto cross_size{mat1_.num_cols()};
    auto val{mat1_(row_index, 0) * mat2_(0, col_index)};
    for (size_t cross_index{1}; cross_index < cross_size; ++cross_index) {
      val += mat1_(row_index, cross_index) * mat2_(cross_index, col_index);
    }
    return val;
  }

}; // class MatrixProductView

template<class Matrix1, class Matrix2>
MatrixProductView(Matrix1&&, Matrix2&&)
    -> MatrixProductView<forward_as_matrix_view_t<Matrix1>,
                         forward_as_matrix_view_t<Matrix2>>;

/// @brief Multiply the matrices @p mat1 and @p mat2.
constexpr auto matmul(viewable_matrix auto&& mat1,
                      viewable_matrix auto&& mat2) noexcept {
  STORM_ASSERT_(mat1.num_cols() == mat2.num_rows() &&
                "The first matrix should have the same number of columns "
                "as the second matrix has rows.");
  return MatrixProductView(std::forward<decltype(mat1)>(mat1),
                           std::forward<decltype(mat2)>(mat2));
}

/// @} // Functional views.

/// @} // Matrix views.

/// @brief Print a @p mat.
std::ostream& operator<<(std::ostream& out, matrix auto&& mat) {
  for (size_t row_index{0}; row_index < mat.num_rows(); ++row_index) {
    out << "( ";
    for (size_t col_index{0}; col_index < mat.num_cols(); ++col_index) {
      out << mat(row_index, col_index) << " ";
    }
    out << ")" << std::endl;
  }
  return out;
}

} // namespace Storm

#include "MatrixAction.hpp"