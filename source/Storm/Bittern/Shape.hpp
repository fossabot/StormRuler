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

#include <Storm/Utils/Meta.hpp>

#include <array>
#include <tuple>

namespace Storm
{

// -----------------------------------------------------------------------------

/// @brief Treat the specified type as the fixed extent.
template<class FixedExtent>
inline constexpr bool enable_fixed_extent_v = false;
template<std::integral Type, Type Value>
inline constexpr bool enable_fixed_extent_v< //
    std::integral_constant<Type, Value>> = true;

/// @brief Fixed extent: integral constant.
template<class FixedExtent>
concept fixed_extent = enable_fixed_extent_v<FixedExtent>;

/// @brief Treat the specified type as the dynamic extent.
template<class DynamicExtent>
inline constexpr bool enable_dynamic_extent_v =
    std::is_integral_v<DynamicExtent>;

/// @brief Dynamic extent: integer.
template<class DynamicExtent>
concept dynamic_extent = enable_dynamic_extent_v<DynamicExtent>;

/// @brief Fixed or dynamic extent.
template<class Extent>
concept extent = true; // fixed_extent<Extent> || dynamic_extent<Extent>;

// -----------------------------------------------------------------------------

/// @brief Shape: a tuple of extents.
template<extent... Extents>
using shape_t = std::tuple<Extents...>;

/// @brief Fixed shape type.
template<size_t... Extents>
using fixed_shape_t = shape_t<size_t_constant<Extents>...>;

/// @brief Dynamic shape type.
template<size_t Rank>
using dyn_shape_t =
    decltype([]<size_t... Extents>(std::index_sequence<Extents...>) {
      return shp(Extents...);
    }(std::make_index_sequence<Rank>{}));

/// @brief Construct a shape.
template<extent... Extents>
constexpr auto shp(Extents... extents) noexcept
{
  return shape_t{extents...};
}

/// @brief Construct a fixed shape.
template<extent auto... FixedExtents>
consteval auto shp() noexcept
{
  return fixed_shape_t<static_cast<size_t>(FixedExtents)...>{};
}

// -----------------------------------------------------------------------------

/// @brief Shape: instantiation of shape_t.
#if 0
template<class Shape>
concept shape =
    requires { std::apply([](auto...) { /*empty*/ }, std::declval<Shape>()); };
#else
template<class Shape>
concept shape = true;
#endif

/// @brief Shape rank.
template<shape Shape>
inline constexpr size_t rank_v = std::tuple_size_v<Shape>;

// -----------------------------------------------------------------------------

template<shape Shape, shape... RestShapes>
constexpr auto common_shape(Shape shape, RestShapes...)
{
  return shape;
}

template<class... Shapes>
using common_shape_t = decltype(common_shape(std::declval<Shapes>()...));

// -----------------------------------------------------------------------------

template<shape... Shapes>
constexpr auto cat_shapes(Shapes... shapes)
{
  return std::tuple_cat(shapes...);
}

template<shape... Shapes>
using cat_shapes_t = decltype(cat_shapes(std::declval<Shapes>()...));

// -----------------------------------------------------------------------------

} // namespace Storm
