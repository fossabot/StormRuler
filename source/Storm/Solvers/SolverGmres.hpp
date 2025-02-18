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

#include <Storm/Solvers/MatrixDense.hpp>
#include <Storm/Solvers/Solver.hpp>

namespace Storm
{

// -----------------------------------------------------------------------------

/// @brief Parmetrized GMRES solver.
template<legacy_vector_like Vector, bool Flexible, bool Loose = false>
class BaseGmresSolver : public InnerOuterIterativeSolver<Vector>
{
private:

  DenseVector<real_t> beta_, cs_, sn_;
  DenseMatrix<real_t> H_;
  std::vector<Vector> q_vecs_;
  std::conditional_t<Flexible, std::vector<Vector>, std::array<Vector, 1>>
      z_vecs_;

  real_t outer_init(const Vector& x_vec, const Vector& b_vec,
                    const Operator<Vector>& lin_op,
                    const Preconditioner<Vector>* pre_op) override
  {
    const size_t m = this->num_inner_iterations;

    beta_.assign(m + 1);
    cs_.assign(m), sn_.assign(m);
    H_.assign(m + 1, m);

    q_vecs_.resize(m + 1);
    for (Vector& q_vec : q_vecs_) q_vec.assign(x_vec, false);
    if (pre_op != nullptr) {
      if constexpr (Flexible) z_vecs_.resize(m);
      for (Vector& z_vec : z_vecs_) z_vec.assign(x_vec, false);
    }

    /// @todo Refactor without duplication a code from
    ///   inner_init method.
    const bool left_pre = (pre_op != nullptr) && (!Flexible) &&
                          (this->pre_side == PreconditionerSide::Left);

    // Initialize:
    // ----------------------
    // 𝒒₀ ← 𝒃 - 𝓐𝒙,
    // 𝗶𝗳 𝘓𝘦𝘧𝘵𝘗𝘳𝘦:
    //   𝒛₀ ← 𝒒₀,
    //   𝒒₀ ← 𝓟𝒛₀,
    // 𝗲𝗻𝗱 𝗶𝗳
    // 𝛽₀ ← ‖𝒒₀‖,
    // 𝒒₀ ← 𝒒₀/𝛽₀.
    // ----------------------
    lin_op.Residual(q_vecs_[0], b_vec, x_vec);
    if (left_pre) {
      std::swap(z_vecs_[0], q_vecs_[0]);
      pre_op->mul(q_vecs_[0], z_vecs_[0]);
    }
    beta_(0) = norm_2(q_vecs_[0]);
    q_vecs_[0] /= beta_(0);

    return beta_(0);
  }

  void inner_init(const Vector& x_vec, const Vector& b_vec,
                  const Operator<Vector>& lin_op,
                  const Preconditioner<Vector>* pre_op) override
  {
    // Force right preconditioning for the flexible GMRES.
    const bool left_pre = (pre_op != nullptr) && (!Flexible) &&
                          (this->pre_side == PreconditionerSide::Left);

    // Initialize:
    // ----------------------
    // 𝒒₀ ← 𝒃 - 𝓐𝒙,
    // 𝗶𝗳 𝘓𝘦𝘧𝘵𝘗𝘳𝘦:
    //   𝒛₀ ← 𝒒₀,
    //   𝒒₀ ← 𝓟𝒛₀,
    // 𝗲𝗻𝗱 𝗶𝗳
    // 𝛽₀ ← ‖𝒒₀‖,
    // 𝒒₀ ← 𝒒₀/𝛽₀.
    // ----------------------
    lin_op.Residual(q_vecs_[0], b_vec, x_vec);
    if (left_pre) {
      std::swap(z_vecs_[0], q_vecs_[0]);
      pre_op->mul(q_vecs_[0], z_vecs_[0]);
    }
    beta_(0) = norm_2(q_vecs_[0]);
    q_vecs_[0] /= beta_(0);
  }

  real_t inner_iterate(Vector& x_vec, const Vector& b_vec,
                       const Operator<Vector>& lin_op,
                       const Preconditioner<Vector>* pre_op) override
  {
    const size_t k = this->inner_iteration;

    // Force right preconditioning for the flexible GMRES.
    const bool left_pre =
        (pre_op != nullptr) &&
        (!Flexible && (this->pre_side == PreconditionerSide::Left));
    const bool right_pre =
        (pre_op != nullptr) &&
        (Flexible || (this->pre_side == PreconditionerSide::Right));

    // Compute the new 𝒒ₖ₊₁ vector:
    // ----------------------
    // 𝗶𝗳 𝘓𝘦𝘧𝘵𝘗𝘳𝘦:
    //   𝒒ₖ₊₁ ← 𝓟(𝒛₀ ← 𝓐𝒒ₖ),
    // 𝗲𝗹𝘀𝗲 𝗶𝗳 𝘙𝘪𝘨𝘩𝘵𝘗𝘳𝘦:
    //   𝑗 ← 𝘍𝘭𝘦𝘹𝘪𝘣𝘭𝘦 ? 𝑘 : 𝟢,
    //   𝒒ₖ₊₁ ← 𝓐(𝒛ⱼ ← 𝓟𝒒ₖ),
    // 𝗲𝗹𝘀𝗲:
    //   𝒒ₖ₊₁ ← 𝓐𝒒ₖ,
    // 𝗲𝗻𝗱 𝗶𝗳
    // 𝗳𝗼𝗿 𝑖 = 𝟢, 𝑘 𝗱𝗼:
    //   𝐻ᵢₖ ← <𝒒ₖ₊₁⋅𝒒ᵢ>,
    //   𝒒ₖ₊₁ ← 𝒒ₖ₊₁ - 𝐻ᵢₖ⋅𝒒ᵢ,
    // 𝗲𝗻𝗱 𝗳𝗼𝗿
    // 𝐻ₖ₊₁,ₖ ← ‖𝒒ₖ₊₁‖,
    // 𝒒ₖ₊₁ ← 𝒒ₖ₊₁/𝐻ₖ₊₁,ₖ.
    // ----------------------
    if (left_pre) {
      pre_op->mul(q_vecs_[k + 1], z_vecs_[0], lin_op, q_vecs_[k]);
    } else if (right_pre) {
      const size_t j{Flexible ? k : 0};
      lin_op.mul(q_vecs_[k + 1], z_vecs_[j], *pre_op, q_vecs_[k]);
    } else {
      lin_op.mul(q_vecs_[k + 1], q_vecs_[k]);
    }
    for (size_t i = 0; i <= k; ++i) {
      H_(i, k) = dot_product(q_vecs_[k + 1], q_vecs_[i]);
      q_vecs_[k + 1] -= H_(i, k) * q_vecs_[i];
    }
    H_(k + 1, k) = norm_2(q_vecs_[k + 1]);
    q_vecs_[k + 1] /= H_(k + 1, k);

    // Eliminate the last element in 𝐻
    // and and update the rotation matrix:
    // ----------------------
    // 𝗳𝗼𝗿 𝑖 = 𝟢, 𝑘 - 𝟣 𝗱𝗼:
    //   𝜒 ← 𝑐𝑠ᵢ⋅𝐻ᵢₖ + 𝑠𝑛ᵢ⋅𝐻ᵢ₊₁,ₖ,
    //   𝐻ᵢ₊₁,ₖ ← -𝑠𝑛ᵢ⋅𝐻ᵢₖ + 𝑐𝑠ᵢ⋅𝐻ᵢ₊₁,ₖ,
    //   𝐻ᵢₖ ← 𝜒,
    // 𝗲𝗻𝗱 𝗳𝗼𝗿
    // 𝑐𝑠ₖ, 𝑠𝑛ₖ ← 𝘚𝘺𝘮𝘖𝘳𝘵𝘩𝘰(𝐻ₖₖ, 𝐻ₖ₊₁,ₖ),
    // 𝐻ₖₖ ← 𝑐𝑠ₖ⋅𝐻ₖₖ + 𝑠𝑛ₖ⋅𝐻ₖ₊₁,ₖ,
    // 𝐻ₖ₊₁,ₖ ← 𝟢.
    // ----------------------
    for (size_t i = 0; i < k; ++i) {
      const real_t chi = cs_(i) * H_(i, k) + sn_(i) * H_(i + 1, k);
      H_(i + 1, k) = -sn_(i) * H_(i, k) + cs_(i) * H_(i + 1, k);
      H_(i, k) = chi;
    }
    std::tie(cs_(k), sn_(k), std::ignore) = sym_ortho(H_(k, k), H_(k + 1, k));
    H_(k, k) = cs_(k) * H_(k, k) + sn_(k) * H_(k + 1, k);
    H_(k + 1, k) = 0.0;

    // Update the 𝛽-solution and the residual norm:
    // ----------------------
    // 𝛽ₖ₊₁ ← -𝑠𝑛ₖ⋅𝛽ₖ, 𝛽ₖ ← 𝑐𝑠ₖ⋅𝛽ₖ.
    // ----------------------
    beta_(k + 1) = -sn_(k) * beta_(k), beta_(k) *= cs_(k);

    return std::abs(beta_(k + 1));
  }

  void inner_finalize(Vector& x_vec, const Vector& b_vec,
                      const Operator<Vector>& lin_op,
                      const Preconditioner<Vector>* pre_op) override
  {
    const size_t k = this->inner_iteration;

    const bool right_pre =
        (pre_op != nullptr) &&
        (Flexible || (this->pre_side == PreconditionerSide::Right));

    // Finalize the 𝛽-solution:
    // ----------------------
    // 𝛽₀:ₖ ← (𝐻₀:ₖ,₀:ₖ)⁻¹𝛽₀:ₖ.
    // ----------------------
    for (size_t i = k; i != SIZE_MAX; --i) {
      for (size_t j{i + 1}; j <= k; ++j) {
        beta_(i) -= H_(i, j) * beta_(j);
      }
      beta_(i) /= H_(i, i);
    }

    // Compute the 𝒙-solution:
    // ----------------------
    // 𝗶𝗳 𝗻𝗼𝘁 𝘙𝘪𝘨𝘩𝘵𝘗𝘳𝘦:
    //   𝗳𝗼𝗿 𝑖 = 𝟢, 𝑘 𝗱𝗼:
    //     𝒙 ← 𝒙 + 𝛽ᵢ⋅𝒒ᵢ.
    //   𝗲𝗻𝗱 𝗳𝗼𝗿
    // 𝗲𝗹𝘀𝗲 𝗶𝗳 𝘍𝘭𝘦𝘹𝘪𝘣𝘭𝘦:
    //   𝗳𝗼𝗿 𝑖 = 𝟢, 𝑘 𝗱𝗼:
    //     𝒙 ← 𝒙 + 𝛽ᵢ⋅𝒛ᵢ.
    //   𝗲𝗻𝗱 𝗳𝗼𝗿
    // 𝗲𝗹𝘀𝗲:
    //   𝒒₀ ← 𝛽₀⋅𝒒₀,
    //   𝗳𝗼𝗿 𝑖 = 𝟣, 𝑘 𝗱𝗼:
    //     𝒒₀ ← 𝒒₀ + 𝛽ᵢ⋅𝒒ᵢ,
    //   𝗲𝗻𝗱 𝗳𝗼𝗿
    //   𝒛₀ ← 𝓟𝒒₀,
    //   𝒙 ← 𝒙 + 𝒛₀.
    // 𝗲𝗻𝗱 𝗶𝗳
    // ----------------------
    if (!right_pre) {
      for (size_t i = 0; i <= k; ++i) {
        x_vec += beta_(i) * q_vecs_[i];
      }
    } else if constexpr (Flexible) {
      for (size_t i = 0; i <= k; ++i) {
        x_vec += beta_(i) * z_vecs_[i];
      }
    } else {
      q_vecs_[0] *= beta_(0);
      for (size_t i = 1; i <= k; ++i) {
        q_vecs_[0] += beta_(i) * q_vecs_[i];
      }
      pre_op->mul(z_vecs_[0], q_vecs_[0]);
      x_vec += z_vecs_[0];
    }
  }

protected:

  BaseGmresSolver() = default;

}; // class BaseGmresSolver

// -----------------------------------------------------------------------------

/// @brief The GMRES (Generalized Minimal Residual) linear operator equation
/// solver.
///
/// GMRES is typically more robust than the BiCG type solvers, but it may be
/// slower than the BiCG solvers for the well-conditioned moderate sized
/// problems.
///
/// GMRES is algebraically equivalent to MINRES method in the self-adjoint
/// operator unpreconditioned case, however, the need for restarts may lead to
/// the much slower GMRES convergence rate.
///
/// GMRES may be applied to the singular problems, and the square least squares
/// problems, although, similarly to MINRES, convergeance to minimum norm
/// solution is not guaranteed.
///
/// References:
/// @verbatim
/// [1] Saad, Yousef and Martin H. Schultz.
///     “GMRES: A generalized minimal residual algorithm for solving
///      nonsymmetric linear systems.”
///     SIAM J. Sci. Stat. Comput., 7:856–869, 1986.
/// @endverbatim
template<legacy_vector_like Vector>
class GmresSolver final : public BaseGmresSolver<Vector, false>
{
}; // class GmresSolver

/// @brief The FGMRES (Flexible Generalized Minimal Residual) linear operator
/// equation solver.
///
/// FGMRES is typically more robust than the BiCG type solvers, but it may be
/// slower than the BiCG solvers for the well-conditioned moderate sized
/// problems.
///
/// FGMRES does the same amount of operations per iteration as GMRES, but also
/// allows usage of the variable (or flexible) preconditioners with the price of
/// doubleing of the memory usage. For the static preconditioners, FGMRES
/// requires one preconditioner-vector product less than GMRES.
/// FGMRES supports only the right preconditioning.
///
/// FGMRES may be applied to the singular problems, and the square least squares
/// problems, although, similarly to MINRES, convergeance to minimum norm
/// solution is not guaranteed.
///
/// References:
/// @verbatim
/// [1] Saad, Yousef.
///     “A Flexible Inner-Outer Preconditioned GMRES Algorithm.”
///     SIAM J. Sci. Comput. 14 (1993): 461-469.
/// @endverbatim
template<legacy_vector_like Vector>
class FgmresSolver final : public BaseGmresSolver<Vector, true>
{
}; // class FgmresSolver

// -----------------------------------------------------------------------------

} // namespace Storm
