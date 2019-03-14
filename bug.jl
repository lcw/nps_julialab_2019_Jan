using LinearAlgebra
using Random
using GPUifyLoops
using StaticArrays

const HAVE_CUDA = try
    using CUDAdrv
    using CUDAnative
    using CuArrays
    true
catch
    false
end
if !HAVE_CUDA
    macro cuStaticSharedMem(x...)
        :()
    end
    macro cuda(x...)
        :()
    end
end

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

# {{{ Volume RHS for 3-D
function volumerhs_v1!(::Val{3}, ::Val{N}, ::Val{nmoist}, ::Val{ntrace},
                       rhs::Array, Q, vgeo, gravity, D,
                       elems) where {N, nmoist, ntrace}
    DFloat = eltype(Q)

    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    nelem = size(Q)[end]

    Q = reshape(Q, Nq, Nq, Nq, nvar, nelem)
    rhs = reshape(rhs, Nq, Nq, Nq, nvar, nelem)
    vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    l_u = Array{DFloat}(undef, Nq, Nq, Nq)
    l_v = Array{DFloat}(undef, Nq, Nq, Nq)
    l_w = Array{DFloat}(undef, Nq, Nq, Nq)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            MJI = vgeo[i, j, k, _MJI, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

            P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

            ρinv = 1 / ρ
            fluxρ_x = U
            fluxU_x = ρinv * U * U  + P
            fluxV_x = ρinv * U * V
            fluxW_x = ρinv * U * W
            fluxE_x = ρinv * U * (E + P)

            fluxρ_y = V
            fluxU_y = ρinv * V * U
            fluxV_y = ρinv * V * V + P
            fluxW_y = ρinv * V * W
            fluxE_y = ρinv * V * (E + P)

            fluxρ_z = W
            fluxU_z = ρinv * W * U
            fluxV_z = ρinv * W * V
            fluxW_z = ρinv * W * W + P
            fluxE_z = ρinv * W * (E + P)

            s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
            s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
            s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
            s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
            s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

            s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
            s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
            s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
            s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
            s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

            s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
            s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
            s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
            s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
            s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

            # buoyancy term
            rhs[i, j, k, _W, e] -= ρ * gravity

            # Store velocity
            l_u[i, j, k], l_v[i, j, k], l_w[i, j, k] = ρinv * U, ρinv * V, ρinv * W
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJI = vgeo[i, j, k, _MJI, e]
            for n = 1:Nq
                rhs[i, j, k, s, e] += MJI * D[n, i] * s_F[n, j, k, s]
            end
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJI = vgeo[i, j, k, _MJI, e]
            for n = 1:Nq
                rhs[i, j, k, s, e] += MJI * D[n, j] * s_G[i, n, k, s]
            end
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJI = vgeo[i, j, k, _MJI, e]
            for n = 1:Nq
                rhs[i, j, k, s, e] += MJI * D[n, k] * s_H[i, j, n, s]
            end
        end

        # loop over moist variables
        # FIXME: Currently just passive advection
        for m = 1:nmoist
            s = _nstate + m

            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJ = vgeo[i, j, k, _MJ, e]
                ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
                u, v, w = l_u[i, j, k], l_v[i, j, k], l_w[i, j, k]

                fx = u * Q[i, j, k, s, e]
                fy = v * Q[i, j, k, s, e]
                fz = w * Q[i, j, k, s, e]

                s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, i] * s_F[n, j, k, 1]
                end
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, j] * s_G[i, n, k, 1]
                end
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, k] * s_H[i, j, n, 1]
                end
            end
        end

        # loop over tracer variables
        for t = 1:ntrace
            s = _nstate + nmoist + t

            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJ = vgeo[i, j, k, _MJ, e]
                ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
                u, v, w = l_u[i, j, k], l_v[i, j, k], l_w[i, j, k]
                
                fx = u * Q[i, j, k, s, e]
                fy = v * Q[i, j, k, s, e]
                fz = w * Q[i, j, k, s, e]

                s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, i] * s_F[n, j, k, 1]
                end
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, j] * s_G[i, n, k, 1]
                end
            end
            for k = 1:Nq, j = 1:Nq, i = 1:Nq
                MJI = vgeo[i, j, k, _MJI, e]
                for n = 1:Nq
                    rhs[i, j, k, s, e] += MJI * D[n, k] * s_H[i, j, n, 1]
                end
            end
        end
end
end
# }}}

# {{{ volume_v5
function volumerhs_v5!(::Val{3},
                       ::Val{N},
                       ::Val{nmoist},
                       ::Val{ntrace},
                       rhs,
                       Q,
                       vgeo,
                       gravity,
                       D,
                       nelem) where {N, nmoist, ntrace}

  nvar = _nstate + nmoist + ntrace

  Nq = N + 1

  s_D = @cuStaticSharedMem eltype(D) (Nq, Nq)
  s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)
  s_G = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)
  s_H = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)

  r_rhsρ = MArray{Tuple{Nq},eltype(rhs)}(undef)
  r_rhsU = MArray{Tuple{Nq},eltype(rhs)}(undef)
  r_rhsV = MArray{Tuple{Nq},eltype(rhs)}(undef)
  r_rhsW = MArray{Tuple{Nq},eltype(rhs)}(undef)
  r_rhsE = MArray{Tuple{Nq},eltype(rhs)}(undef)

  e = blockIdx().x
  j = threadIdx().y
  i = threadIdx().x

  @inbounds begin
  for k in 1:Nq
    r_rhsρ[k] = zero(eltype(rhs))
    r_rhsU[k] = zero(eltype(rhs))
    r_rhsV[k] = zero(eltype(rhs))
    r_rhsW[k] = zero(eltype(rhs))
    r_rhsE[k] = zero(eltype(rhs))
  end

  # fetch D into shared
  s_D[i, j] = D[i, j]

  for k in 1:Nq
    sync_threads()

    # Load values will need into registers
    MJ = vgeo[i, j, k, _MJ, e]
    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
    z = vgeo[i,j,k,_z,e]

    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

    ρinv = 1 / ρ

    fluxρ_x = U
    fluxU_x = ρinv * U * U + P
    fluxV_x = ρinv * U * V
    fluxW_x = ρinv * U * W
    fluxE_x = ρinv * U * (E + P)

    fluxρ_y = V
    fluxU_y = ρinv * V * U
    fluxV_y = ρinv * V * V + P
    fluxW_y = ρinv * V * W
    fluxE_y = ρinv * V * (E + P)

    fluxρ_z = W
    fluxU_z = ρinv * W * U
    fluxV_z = ρinv * W * V
    fluxW_z = ρinv * W * W + P
    fluxE_z = ρinv * W * (E + P)

    s_F[i, j,  _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
    s_F[i, j,  _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
    s_F[i, j,  _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
    s_F[i, j,  _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
    s_F[i, j,  _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

    s_G[i, j,  _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
    s_G[i, j,  _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
    s_G[i, j,  _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
    s_G[i, j,  _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
    s_G[i, j,  _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

    r_Hρ = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
    r_HU = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
    r_HV = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
    r_HW = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
    r_HE = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

    # one shared access per 10 flops
    for n = 1:Nq
      Dkn = s_D[k, n]

      r_rhsρ[n] += Dkn * r_Hρ
      r_rhsU[n] += Dkn * r_HU
      r_rhsV[n] += Dkn * r_HV
      r_rhsW[n] += Dkn * r_HW
      r_rhsE[n] += Dkn * r_HE
    end

    r_rhsW[k,i,j] -= MJ * ρ * gravity

    sync_threads()
    # loop of ξ-grid lines
    for n = 1:Nq
      Dni = s_D[n, i]
      Dnj = s_D[n, j]

      r_rhsρ[k] += Dni * s_F[n, j, _ρ]
      r_rhsρ[k] += Dnj * s_G[i, n, _ρ]

      r_rhsU[k] += Dni * s_F[n, j, _U]
      r_rhsU[k] += Dnj * s_G[i, n, _U]

      r_rhsV[k] += Dni * s_F[n, j, _V]
      r_rhsV[k] += Dnj * s_G[i, n, _V]

      r_rhsW[k] += Dni * s_F[n, j, _W]
      r_rhsW[k] += Dnj * s_G[i, n, _W]

      r_rhsE[k] += Dni * s_F[n, j, _E]
      r_rhsE[k] += Dnj * s_G[i, n, _E]
    end
  end # k

  for k in 1:Nq
    MJI = vgeo[i, j, k, _MJI, e]

    rhs[i, j, k, _U, e] += MJI*r_rhsU[k]
    rhs[i, j, k, _V, e] += MJI*r_rhsV[k]
    rhs[i, j, k, _W, e] += MJI*r_rhsW[k]
    rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k]
    rhs[i, j, k, _E, e] += MJI*r_rhsE[k]
  end
  end
  nothing
end
# }}}

function main(nelem, N, DFloat)
  rnd = MersenneTwister(0)
  nmoist = 0
  ntrace = 0
  nvar = _nstate + nmoist + ntrace

  Nq = N + 1
  Q = 1 .+ rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem)
  Q[:, :, :, _E, :] .+= 20
  vgeo = rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem)

  # Make sure the entries of the mass matrix satisfy the inverse relation
  vgeo[:, :, :, _MJ, :] .+= 3
  vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

  D = rand(rnd, DFloat, Nq, Nq)

  rhs_v1 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs_v1!(Val(3), Val(N), Val(nmoist), Val(ntrace), rhs_v1, Q, vgeo,
                DFloat(grav), D, 1:nelem)
  @show norm_v1 = norm(rhs_v1)

  if HAVE_CUDA
    d_Q = CuArray(Q)
    d_D = CuArray(D)
    d_vgeo = CuArray(vgeo)

    rhs_v5 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
    d_rhs_v5 = CuArray(rhs_v5)

    @device_code dir="bug" @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs_v5!(Val(3), Val(N), Val(nmoist), Val(ntrace),
                        d_rhs_v5, d_Q, d_vgeo, DFloat(grav), d_D, nelem))
    rhs_v5 .= d_rhs_v5
    @show norm_v5 = norm(rhs_v5)
    @show norm_v1 - norm_v5

    @show CUDAdrv.@elapsed @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs_v5!(Val(3), Val(N), Val(nmoist), Val(ntrace), d_rhs_v5,
                        d_Q, d_vgeo, DFloat(grav), d_D, nelem))
  end

  nothing
end

main(4000,4,Float32)
