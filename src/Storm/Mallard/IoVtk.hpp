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

#pragma once

#include <Storm/Base.hpp>

#include <Storm/Utils/Meta.hpp>

#include <Storm/Mallard/Entity.hpp>
#include <Storm/Mallard/Mesh.hpp>
#include <Storm/Mallard/Shape.hpp>

#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <type_traits>

namespace Storm {

template<mesh Mesh>
void write_mesh_to_vtk(const Mesh& mesh, const std::filesystem::path& path) {
  std::ofstream file(path);
  file << std::setprecision(std::numeric_limits<real_t>::digits10 + 1);
  file << "# vtk DataFile Version 3.0\n";
  file << "# Generated by Storm\n";
  file << "ASCII\n";
  file << "DATASET UNSTRUCTURED_GRID\n";

  file << "POINTS " << nodes(mesh).size() << " double\n";
  for (NodeView node : nodes(mesh)) {
    const auto pos = node.position();
    file << pos.x << " " << pos.y << " " << 0.0 /*pos.z*/ << "\n";
  }
  file << "\n";

  size_t const sumNumCellAdjNodes = 4 * cells(mesh).size();
  // ForEachSum(int_cell_views(*this), size_t(0), [](CellView cell) {
  //   return cell.adjacent_nodes().size() + 1;
  // });
  file << "CELLS " << cells(mesh).size() << " " << sumNumCellAdjNodes << "\n";
  for (CellView cell : cells(mesh)) {
    file << cell.nodes().size() << " ";
    cell.for_each_node(
        [&](NodeView<const Mesh> node) { file << node.index() << " "; });
    file << "\n";
  }
  file << std::endl;

  file << "CELL_TYPES " << cells(mesh).size() << "\n";
  for (CellView cell : cells(mesh)) {
    static const std::map<shapes::Type, const char*> shapes{
        {shapes::Type::segment, "2"},        {shapes::Type::triangle, "5"},
        {shapes::Type::quadrangle, "9"},     {shapes::Type::polygon, "7"},
        {shapes::Type::triangle_strip, "6"}, {shapes::Type::tetrahedron, "10"},
        {shapes::Type::pyramid, "14"},       {shapes::Type::pentahedron, "13"},
        {shapes::Type::hexahedron, "12"},
    };
    file << shapes.at(cell.shape_type()) << "\n";
  }
  file << "\n";

#if 0
  file << "CELL_DATA " << cells({}).size() << std::endl;
  for (const sFieldDesc& field : fields) {
    file << "SCALARS " << field.name << " double 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    ranges::for_each(int_cell_views(*this), [&](CellView cell) {
      file << (*field.scalar)[cell][field.var_index] << std::endl;
    });
  }
  file << std::endl;
#endif
} // Mesh::save_vtk

} // namespace Storm