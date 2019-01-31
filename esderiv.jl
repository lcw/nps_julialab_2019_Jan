using LinearAlgebra
using Random
using StaticArrays

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                 x = _x,   y = _y,   z = _z)
# }}}

Base.@irrational grav 9.81 BigFloat(9.81)
Base.@irrational R_d 287.0024093890231 BigFloat(287.0024093890231)
Base.@irrational cp_d 1004.5084328615809 BigFloat(1004.5084328615809)
Base.@irrational cv_d 717.5060234725578 BigFloat(717.5060234725578)
Base.@irrational gamma_d 1.4 BigFloat(1.4)
Base.@irrational gdm1 0.4 BigFloat(0.4)


function aln(L, R)
  ζ = L / R
  f = (ζ - 1) / (ζ + 1)
  u = f * f
  ϵ = eltype(L)(1e-2)
  F = (u < ϵ) ?  F = 1 + u / 3 + u^2 / 5 + u^3 / 7 : log(ζ) / (2f)
  (L + R) / 2F
end

function flux!(F, UM, VM, WM, ρM, EM, zM, UP, VP, WP, ρP, EP, zP, gravity)
  ρMinv, ρPinv = 1 / ρM, 1/ρP
  uM, vM, wM = UM * ρMinv, VM * ρMinv, WM * ρMinv
  uP, vP, wP = UP * ρPinv, VP * ρPinv, WP * ρPinv

  PM = gdm1*(EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM*gravity*zM)
  PP = gdm1*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP*gravity*zP)
  βM = ρM / 2PM
  βP = ρP / 2PP

  ua  = (uM + uP) / 2
  va  = (vM + vP) / 2
  wa  = (wM + wP) / 2
  u2a = ((uM^2 + vM^2 + wM^2) + (uP^2 + vP^2 + wP^2)) / 2
  ρa  = (ρM + ρP) / 2
  βa = (βM + βP) / 2
  ρln = aln(ρM, ρP)
  βln = aln(βM, βP)
  ϕa = gravity * (zM + zP) / 2

  F[_ρ, 1] = ρln * ua
  F[_ρ, 2] = ρln * va
  F[_ρ, 3] = ρln * wa

  F[_U, 1] = F[_ρ, 1] * ua + ρa / 2βa
  F[_V, 1] = F[_ρ, 1] * va
  F[_W, 1] = F[_ρ, 1] * wa

  F[_U, 2] = F[_ρ, 2] * ua
  F[_V, 2] = F[_ρ, 2] * va + ρa / 2βa
  F[_W, 2] = F[_ρ, 2] * wa

  F[_U, 3] = F[_ρ, 3] * ua
  F[_V, 3] = F[_ρ, 3] * va
  F[_W, 3] = F[_ρ, 3] * wa + ρa / 2βa

  Efac = 1 / (2*gdm1*βln) - u2a / 2 + ϕa
  F[_E, 1] = (F[_U, 1] * ua + F[_V, 1] * va + F[_W, 1] * wa) + Efac * F[_ρ, 1]
  F[_E, 2] = (F[_U, 2] * ua + F[_V, 2] * va + F[_W, 2] * wa) + Efac * F[_ρ, 2]
  F[_E, 3] = (F[_U, 3] * ua + F[_V, 3] * va + F[_W, 3] * wa) + Efac * F[_ρ, 3]
end

function flux(q, z, gravity)
  F = similar(q, size(q, 1), size(q, 1), size(q, 2), 3)
  G = similar(q, size(q, 2), 3)
  for n = 1:size(q,1)
    for m = 1:size(q,1)
      flux!(G, q[n, _U], q[n, _V], q[n, _W], q[n, _ρ], q[n, _E], z[n],
            q[m, _U], q[m, _V], q[m, _W], q[m, _ρ], q[m, _E], z[m], gravity)
      F[n, m, :, :] = G[:, :]
    end
  end
  F
end

function _flux_v2(q, z, gravity)
    q_v2 = (ρ = q[:, _ρ], U = map(SVector{3}, q[:, _U], q[:, _V], q[:, _W]), E = q[:, _E])
    F_v2 = flux_v2(q_v2, z, gravity)
    F = similar(q, size(q, 1), size(q, 1), size(q, 2), 3)
    for n = 1:size(q, 1)
          for m = 1:size(q, 1)
              #const _ρ, _U, _V, _W, _E = 1:_nstate
              F[n, m, 1, :] = F_v2.ρ[n, m]
              F[n, m, 2:4, :] = F_v2.U[n, m]
              F[n, m, 5, :] = F_v2.E[n, m]
          end
    end
    return F
end



function flux_v2(q, z, gravity)
  F = (ρ = similar(q.ρ, SVector{3, eltype(q.ρ)}, size(q.U, 1), size(q.U, 1)),
       U = similar(q.U, SMatrix{3, 3, eltype(eltype(q.U))}, size(q.U, 1), size(q.U, 1)),
       E = similar(q.E, SVector{3, eltype(q.E)}, size(q.U, 1), size(q.U, 1)))
  for n = eachindex(q.U)
    for m = eachindex(q.U)
      (F.ρ[n, m], F.U[n, m], F.E[n, m]) =
        flux_v2!((ρ = q.ρ[n], U = q.U[n], E = q.E[n], z = z[n]),
                 (ρ = q.ρ[m], U = q.U[m], E = q.E[m], z = z[m]), gravity)
    end
  end
  F
end

function flux_v2!(M, P, gravity)
  (M, P) = map((M, P)) do q
    u = q.U/q.ρ
    P = gdm1*(q.E - q.U'*q.U/(2q.ρ) - q.ρ*gravity*q.z)
    β = q.ρ / 2P
    q = (q..., u=u, P=P, β = β)
  end

  ua = (M.u + P.u) / 2
  u2a = (M.u'*M.u + P.u'*P.u) / 2
  ρa  = (M.ρ + P.ρ) / 2
  βa = (M.β + P.β) / 2
  ρln = aln(M.ρ, P.ρ)
  βln = aln(M.β, P.β)
  ϕa = gravity * (M.z + P.z) / 2

  ρ = ρln * ua
  U = ρ * ua' + I * ρa/2βa
  E = U' * ua + ρ * (1 / (2*gdm1*βln) - u2a / 2 + ϕa)
  F = (ρ, U, E)
end


# {{{ Volume RHS for 3-D
function volumerhs!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N}
  nvar = _nstate

  DFloat = eltype(Q)

  Nq = N + 1

  nelem = last(size(Q))

  F       = Array{DFloat}(undef, _nstate, 3)
  l_MJrhs = Array{DFloat}(undef, _nstate)

  @inbounds for e in elems
    for k = 1:Nq, j = 1:Nq, i = 1:Nq

      #{{{ loads for ijk
      ρijk = Q[i, j, k, _ρ, e]
      Uijk = Q[i, j, k, _U, e]
      Vijk = Q[i, j, k, _V, e]
      Wijk = Q[i, j, k, _W, e]
      Eijk = Q[i, j, k, _E, e]

      MJijk = vgeo[i, j, k, _MJ, e]
      ξxijk = vgeo[i, j, k, _ξx, e]
      ξyijk = vgeo[i, j, k, _ξy, e]
      ξzijk = vgeo[i, j, k, _ξz, e]
      ηxijk = vgeo[i, j, k, _ηx, e]
      ηyijk = vgeo[i, j, k, _ηy, e]
      ηzijk = vgeo[i, j, k, _ηz, e]
      ζxijk = vgeo[i, j, k, _ζx, e]
      ζyijk = vgeo[i, j, k, _ζy, e]
      ζzijk = vgeo[i, j, k, _ζz, e]
      yorzijk = vgeo[i, j, k, _z, e]
      #}}}

      l_MJrhs  .= 0
      for n = 1:Nq
        # ξ-direction
        #{{{ loads for njk
        #
        ρnjk = Q[n, j, k, _ρ, e]
        Unjk = Q[n, j, k, _U, e]
        Vnjk = Q[n, j, k, _V, e]
        Wnjk = Q[n, j, k, _W, e]
        Enjk = Q[n, j, k, _E, e]

        MJnjk = vgeo[n, j, k, _MJ, e]
        ξxnjk = vgeo[n, j, k, _ξx, e]
        ξynjk = vgeo[n, j, k, _ξy, e]
        ξznjk = vgeo[n, j, k, _ξz, e]
        yorznjk = vgeo[n, j, k, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Unjk, Vnjk, Wnjk, ρnjk, Enjk, yorznjk,
                 gravity)
        for s = 1:_nstate
          # J W ξ_{x} (D_ξ ∘ F_{sx}) 1⃗ + J W ξ_{y} (D_ξ ∘ F_{sy}) 1⃗ +
          # J W ξ_{z} (D_ξ ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ξxijk * D[i, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ξyijk * D[i, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ξzijk * D[i, n] * F[s, 3]
          # (F_{sx} ∘ D_ξ^T) J W ξ_{x} 1⃗ + (F_{sy} ∘ D_ξ^T) J W ξ_{y} 1⃗ +
          # (F_{sz} ∘ D_ξ^T) J W ξ_{z} 1⃗
          l_MJrhs[s] -= D[n, i] * F[s, 1] * MJnjk * ξxnjk
          l_MJrhs[s] -= D[n, i] * F[s, 2] * MJnjk * ξynjk
          l_MJrhs[s] -= D[n, i] * F[s, 3] * MJnjk * ξznjk
        end

        # η-direction
        #{{{ loads for ink
        ρink = Q[i, n, k, _ρ, e]
        Uink = Q[i, n, k, _U, e]
        Vink = Q[i, n, k, _V, e]
        Wink = Q[i, n, k, _W, e]
        Eink = Q[i, n, k, _E, e]

        MJink = vgeo[i, n, k, _MJ, e]
        ηxink = vgeo[i, n, k, _ηx, e]
        ηyink = vgeo[i, n, k, _ηy, e]
        ηzink = vgeo[i, n, k, _ηz, e]
        yorzink = vgeo[i, n, k, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Uink, Vink, Wink, ρink, Eink, yorzink,
                 gravity)
        for s = 1:_nstate
          # J W η_{x} (D_η ∘ F_{sx}) 1⃗ + J W η_{y} (D_η ∘ F_{sy}) 1⃗ +
          # J W η_{z} (D_η ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ηxijk * D[j, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ηyijk * D[j, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ηzijk * D[j, n] * F[s, 3]
          # (F_{sx} ∘ D_η^T) J W η_{x} 1⃗ + (F_{sy} ∘ D_η^T) J W η_{y} 1⃗ +
          # (F_{sz} ∘ D_η^T) J W η_{z} 1⃗
          l_MJrhs[s] -= D[n, j] * F[s, 1] * MJink * ηxink
          l_MJrhs[s] -= D[n, j] * F[s, 2] * MJink * ηyink
          l_MJrhs[s] -= D[n, j] * F[s, 3] * MJink * ηzink
        end

        # ζ-direction
        #{{{ loads for ijn
        ρijn = Q[i, j, n, _ρ, e]
        Uijn = Q[i, j, n, _U, e]
        Vijn = Q[i, j, n, _V, e]
        Wijn = Q[i, j, n, _W, e]
        Eijn = Q[i, j, n, _E, e]

        MJijn = vgeo[i, j, n, _MJ, e]
        ζxijn = vgeo[i, j, n, _ζx, e]
        ζyijn = vgeo[i, j, n, _ζy, e]
        ζzijn = vgeo[i, j, n, _ζz, e]
        yorzijn = vgeo[i, j, n, _z, e]
        #}}}

        flux!(F, Uijk, Vijk, Wijk, ρijk, Eijk, yorzijk,
                 Uijn, Vijn, Wijn, ρijn, Eijn, yorzijn,
                 gravity)
        for s = 1:_nstate
          # J W ζ_{x} (D_ζ ∘ F_{sx}) 1⃗ + J W ζ_{y} (D_ζ ∘ F_{sy}) 1⃗ +
          # J W ζ_{z} (D_ζ ∘ F_{sz}) 1⃗
          l_MJrhs[s] += MJijk * ζxijk * D[k, n] * F[s, 1]
          l_MJrhs[s] += MJijk * ζyijk * D[k, n] * F[s, 2]
          l_MJrhs[s] += MJijk * ζzijk * D[k, n] * F[s, 3]
          # (F_{sx} ∘ D_ζ^T) J W ζ_{x} 1⃗ + (F_{sy} ∘ D_ζ^T) J W ζ_{y} 1⃗ +
          # (F_{sz} ∘ D_ζ^T) J W ζ_{z} 1⃗
          l_MJrhs[s] -= D[n, k] * F[s, 1] * MJijn * ζxijn
          l_MJrhs[s] -= D[n, k] * F[s, 2] * MJijn * ζyijn
          l_MJrhs[s] -= D[n, k] * F[s, 3] * MJijn * ζzijn
        end
      end

      MJI = vgeo[i, j, k, _MJI, e]
      for s = 1:_nstate
        rhs[i, j, k, s, e] -= MJI * l_MJrhs[s]
      end

      # FIXME: buoyancy term
      rhs[i, j, k,  _W, e] -= ρijk * gravity
    end
  end
end
# }}}

# {{{ Volume RHS for 3-D
function volumerhs_v2!(::Val{3}, ::Val{N}, rhs::Array, Q, vgeo, gravity, D,
                    elems) where {N}
  nvar = _nstate

  DFloat = eltype(Q)

  Np = (N+1)^3

  Q = reshape(Q, Np, _nstate, last(size(Q)))
  vgeo = reshape(vgeo, Np, _nvgeo, last(size(vgeo)))
  rhs = reshape(rhs, Np, _nstate, last(size(rhs)))

  eye = Array{DFloat}(I, N+1, N+1)
  Drst = (kron(eye, eye, D), kron(eye, D, eye), kron(D, eye, eye))

  @inbounds for e in elems

    F = _flux_v2(Q[:, :, e], vgeo[:, _z, e], gravity)

    G = ((vgeo[:, _ξx, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ξy, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ξz, e] .* vgeo[:, _MJ, e]),
         (vgeo[:, _ηx, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ηy, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ηz, e] .* vgeo[:, _MJ, e]),
         (vgeo[:, _ζx, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ζy, e] .* vgeo[:, _MJ, e],
          vgeo[:, _ζz, e] .* vgeo[:, _MJ, e]))
    MJI = vgeo[:, _MJI, e]

    for s = 1:nvar, ξind = 1:3, xind = 1:3
      rhs[:, s, e] -=
      MJI .* (G[ξind][xind] .* ((Drst[ξind] .* F[:, :, s, xind]) * ones(Np)) -
              (F[:, :, s, xind] .* Drst[ξind]') * G[ξind][xind])
    end

    rhs[:, _W, e] -= Q[:, _ρ, e] * gravity
  end

end
# }}}

function main(nelem, N, DFloat)
  rnd = MersenneTwister(0)
  nvar = _nstate

  Nq = N + 1
  Q = 1 .+ rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem)
  Q[:, :, :, _E, :] .+= 100
  vgeo = rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem)
  D = rand(rnd, DFloat, Nq, Nq)

  rhs_v1 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs!(Val(3), Val(N), rhs_v1, Q, vgeo, DFloat(grav), D, 1:nelem)
  @show norm_v1 = norm(rhs_v1)

  rhs_v2 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs_v2!(Val(3), Val(N), rhs_v2, Q, vgeo, DFloat(grav), D, 1:nelem)
  @show norm_v2 = norm(rhs_v2)
  @show norm_v1 - norm_v2
  @show rhs_v1 ≈ rhs_v2
  nothing
end

main(100, 2, Float64)
