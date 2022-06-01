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

#include <cmath>
#include <array>
#include <type_traits>
#include <iostream>

#include <stormBase.hxx>

namespace Storm {

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @brief Statically-sized matrix.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
template<class Value, size_t SizeX, size_t SizeY>
class Mat;

/// @brief 2x2 matrix.
template<class Value>
using Mat2x2 = Mat<Value, 2, 2>;

/// @brief 3x3 matrix.
template<class Value>
using Mat3x3 = Mat<Value, 3, 3>;

/// @brief 4x4 matrix.
template<class Value>
using Mat4x4 = Mat<Value, 4, 4>;

/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
/// @brief Statically-sized vector.
/// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ///
template<class Value, size_t Size>
using Vec = Mat<Value, Size, 1>;

/// @brief 2D vector.
template<class Value>
using Vec2D = Vec<Value, 2>;

/// @brief 3D vector.
template<class Value>
using Vec3D = Vec<Value, 3>;

/// @brief 4D vector.
template<class Value>
using Vec4D = Vec<Value, 4>;

template<class Value, size_t SizeX, size_t SizeY>
class Mat final {
private:
  std::array<std::array<Value, SizeY>, SizeX> data_;

public:

  /// @brief Default constructor.
  constexpr Mat() = default;

  /// @brief Construct the matrix with the initializer list.
  constexpr Mat(std::initializer_list<Value> initializer) {
    StormAssert(initializer.size() == SizeX * SizeY);
    std::copy(initializer.begin(), initializer.end(), data());
  }

  /// @brief Get pointer to the beginning of the vector data.
  /// @{
  constexpr Value* data() noexcept {
    return data_[0].data();
  }
  constexpr Value const* data() const noexcept {
    return data_[0].data();
  }
  /// @}

  /// @brief Get reference to the component at the index.
  /// @{
  constexpr Value& operator()(size_t ix, size_t iy = 0) noexcept {
    StormAssert(ix < SizeX && iy < SizeY);
    return (data_[ix])[iy];
  }
  constexpr Value const& operator()(size_t ix, size_t iy = 0) const noexcept {
    StormAssert(ix < SizeX && iy < SizeY);
    return (data_[ix])[iy];
  }
  /// @}

}; // class Mat

/// @todo Parse types in order to get the output type.
template<class Value, class... Values>
using ResultType = Value;

/// @brief Make a vector.
template<size_t Size, class Value>
constexpr auto MakeVec(Value const& val) noexcept {
  Vec<Value, Size> vec;
  for (size_t i{0}; i < Size; ++i) {
    vec(i) = val;
  }
  return vec;
}

/// @brief Make a diagonal matrix with value @p val of a diagonal.
template<size_t SizeX, size_t SizeY = SizeX, class Value>
constexpr auto make_mat(Value const& val) noexcept {
  Mat<Value, SizeX, SizeY> mat;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < ix; ++iy) {
      mat(ix, iy) = Value(0.0);
    }
    mat(ix, ix) = val;
    for (size_t iy{ix + 1}; iy < SizeY; ++iy) {
      mat(ix, iy) = Value(0.0);
    }
  }
  return mat;
}

/// @brief Transpose a matrix @p mat.
template<class Value, size_t SizeX, size_t SizeY>
constexpr auto Transpose(Mat<Value, SizeX, SizeY> const& mat) noexcept {
  Mat<Value, SizeY, SizeX> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(iy, ix) = mat(ix, iy);
    }
  }
  return out;
}

/// @brief Copy a matrix @p mat.
template<class Value, size_t SizeX, size_t SizeY>
constexpr auto operator+(Mat<Value, SizeX, SizeY> const& mat) noexcept {
  return mat;
}

/// @brief Negate a matrix @p mat.
template<class Value, size_t SizeX, size_t SizeY>
constexpr auto operator-(Mat<Value, SizeX, SizeY> const& mat) noexcept {
  Mat<ResultType<Value>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = -mat(ix, iy);
    }
  }
  return out;
}

/// @brief Add matrices @p mat1 and @p mat2.
/// @{
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto& operator+=(Mat<Value1, SizeX, SizeY>& mat1,
                           Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat1(ix, iy) += mat2(ix, iy);
    }
  }
  return mat1;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator+(Mat<Value1, SizeX, SizeY> const& mat1,
                         Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat1(ix, iy) + mat2(ix, iy);
    }
  }
  return out;
}
/// @}

/// @brief Subtract matrices @p mat1 and @p mat2.
/// @{
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto& operator-=(Mat<Value1, SizeX, SizeY>& mat1,
                           Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat1(ix, iy) -= mat2(ix, iy);
    }
  }
  return mat1;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator-(Mat<Value1, SizeX, SizeY> const& mat1,
                         Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat1(ix, iy) - mat2(ix, iy);
    }
  }
  return out;
}
/// @}

/// @brief Multiply a matrix @p mat by a scalar @p val.
/// @{
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto& operator*=(Mat<Value1, SizeX, SizeY>& mat,
                           Value2 const& val) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat(ix, iy) *= val;
    }
  }
  return mat;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator*(Mat<Value1, SizeX, SizeY> const& mat,
                         Value2 const& val) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat(ix, iy) * val;
    }
  }
  return out;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator*(Value1 const& val,
                         Mat<Value2, SizeX, SizeY> const& mat) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = val * mat(ix, iy);
    }
  }
  return out;
}
/// @}

/// @brief Multiply matrices @p mat1 and @p mat2 (in component-wise sense).
/// @{
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto& operator*=(Mat<Value1, SizeX, SizeY>& mat1,
                           Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat1(ix, iy) *= mat2(ix, iy);
    }
  }
  return mat1;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator*(Mat<Value1, SizeX, SizeY> const& mat1,
                         Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat1(ix, iy) * mat2(ix, iy);
    }
  }
  return out;
}
/// @}

/// @brief Divide a matrix @p mat by a scalar @p val.
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator/=(Mat<Value1, SizeX, SizeY>& mat,
                          Value2 const& val) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat(ix, iy) /= val;
    }
  }
  return mat;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator/(Mat<Value1, SizeX, SizeY> const& mat,
                         Value2 const& val) noexcept {
  StormAssert(val != Value2{0});
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat(ix, iy) / val;
    }
  }
  return out;
}

/// @brief Divide matrices @p mat1 and @p mat2 (in the component-wise sense).
/// @{
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto& operator/=(Mat<Value1, SizeX, SizeY>& mat1,
                           Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      mat1(ix, iy) /= mat2(ix, iy);
    }
  }
  return mat1;
}
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto operator/(Mat<Value1, SizeX, SizeY> const& mat1,
                         Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeY> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out(ix, iy) = mat1(ix, iy) / mat2(ix, iy);
    }
  }
  return out;
}
/// @}

/// @brief Dot product of matrices @p mat1 and @p mat2 (in the vector sense).
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY>
constexpr auto dot(Mat<Value1, SizeX, SizeY> const& mat1,
                   Mat<Value2, SizeX, SizeY> const& mat2) noexcept {
  auto out = mat1(0, 0) * mat2(0, 0);
  for (size_t iy{1}; iy < SizeY; ++iy) {
    out += mat1(0, iy) * mat2(0, iy);
  }
  for (size_t ix{1}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out += mat1(ix, iy) * mat2(ix, iy);
    }
  }
  return out;
}

/// @brief Frobenius norm of a matrix.
template<class Value, size_t SizeX, size_t SizeY>
constexpr auto norm(Mat<Value, SizeX, SizeY> const& mat) noexcept {
  return std::sqrt(dot(mat, mat));
}

/// @brief Multiply matrices @p mat1 and @p mat2 (in normal sense).
template<class Value1, class Value2,
         size_t SizeX, size_t SizeY, size_t SizeZ>
constexpr auto matmul(Mat<Value1, SizeX, SizeY> const& mat1,
                      Mat<Value2, SizeY, SizeZ> const& mat2) noexcept {
  Mat<ResultType<Value1, Value2>, SizeX, SizeZ> out;
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iz{0}; iz < SizeZ; ++iz) {
      out(ix, iz) = mat1(ix, 0) * mat2(0, iz);
      for (size_t iy{1}; iy < SizeY; ++iy) {
        out(ix, iz) += mat1(ix, iy) * mat2(iy, iz);
      }
    }
  }
  return out;
}

/// @brief Perform a LU decomposition of a square matrix @p mat.
/// @returns A pair of matrices, L and U factors.
template<std::floating_point Value, size_t Size>
constexpr auto decompose_lu(Mat<Value, Size, Size> const& mat,
                            size_t size = Size) noexcept {
  auto l_mat = make_mat<Size>(Value{1});
  auto u_mat = make_mat<Size>(Value{0});
  for (size_t ix{0}; ix < size; ++ix) {
    for (size_t iy{0}; iy < ix; ++iy) {
      l_mat(ix, iy) = mat(ix, iy);
      for (size_t iz{0}; iz < iy; ++iz) {
        l_mat(ix, iy) -= l_mat(ix, iz) * u_mat(iz, iy);
      }
      l_mat(ix, iy) /= u_mat(iy, iy);
    }
    for (size_t iy{ix}; iy < size; ++iy) {
      u_mat(ix, iy) = mat(ix, iy);
      for (size_t iz{0}; iz < ix; ++iz) {
        u_mat(ix, iy) -= l_mat(ix, iz) * u_mat(iz, iy);
      }
    }
  }
  return std::pair(l_mat, u_mat);
}

template<std::floating_point Value, size_t Size>
constexpr void solve_lu(auto& vec,
                        std::pair<Mat<Value, Size, Size>,
                                  Mat<Value, Size, Size>> const& lu,
                        size_t size = Size) {
  auto const& [l_mat, u_mat] = lu;
  for (size_t ix{0}; ix < size; ++ix) {
    for (size_t iy{0}; iy < ix; ++iy) {
      vec(ix) -= l_mat(ix, iy) * vec(iy);
    }
    vec(ix) /= l_mat(ix, ix);
  }
  for (size_t rix{0}; rix < size; ++rix) {
    size_t ix{size - 1 - rix};
    for (size_t iy{ix + 1}; iy < size; ++iy) {
      vec(ix) -= u_mat(ix, iy) * vec(iy);
    }
    vec(ix) /= u_mat(ix, ix);
  }
}

/// @brief Inverse a square matrix @p mat using the LU decomposition.
template<std::floating_point Value, size_t Size>
constexpr auto inverse_lu(Mat<Value, Size, Size> const& mat,
                          size_t size = Size) noexcept {
  auto const lu = decompose_lu(mat, size);
  auto out = make_mat<Size>(Value{1});
  for (size_t iy{0}; iy < size; ++iy) {
    auto out_iy_col = [&](size_t ix) -> Value& { return out(ix, iy); };
    solve_lu(out_iy_col, lu, size);
  }
  return out;
}

/// @brief Perform a QR decomposition of a matrix @p mat.
/// @returns A pair of matrices, Q and R factors.
template<std::floating_point Value, size_t SizeX, size_t SizeY>
constexpr auto decompose_qr(Mat<Value, SizeX, SizeY> const& mat) noexcept {
  Mat<Value, SizeX, SizeY> q_mat;
  auto r_mat = make_mat<SizeY, SizeY>(Value{0});
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      std::abort();
    }
  }
  return std::pair(q_mat, r_mat);
}

/// @brief Print a matrix.
template<class Value, size_t SizeX, size_t SizeY>
std::ostream& operator<<(std::ostream& out,
                         Mat<Value, SizeX, SizeY> const& mat) {
  for (size_t ix{0}; ix < SizeX; ++ix) {
    for (size_t iy{0}; iy < SizeY; ++iy) {
      out << mat(ix, iy) << ' ';
    }
    out << std::endl;
  }
  return out;
}

} // namespace Storm