using LinearAlgebra
using Random
using GPUifyLoops

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

# {{{ Volume RHS for 3-D
function volumerhs_v2!(::Val{DEV}, ::Val{3}, ::Val{N}, ::Val{nmoist},
                       ::Val{ntrace}, rhs, Q, vgeo, gravity, D,
                       nelem) where {DEV, N, nmoist, ntrace}
    @setup DEV

    DFloat = eltype(Q)

    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    s_D = @shmem DFloat (Nq, Nq)
    s_F = @shmem DFloat (Nq, Nq, Nq, _nstate)
    s_G = @shmem DFloat (Nq, Nq, Nq, _nstate)
    s_H = @shmem DFloat (Nq, Nq, Nq, _nstate)
    l_ρinv = @shmem DFloat (Nq, Nq, Nq) 

    @inbounds @loop for e in (1:nelem; blockIdx().x)
        @loop for k in (1:Nq; threadIdx().z)
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)

                    if k == 1
                        s_D[i, j] = D[i, j]
                    end

                    # Load values will need into registers
                    MJ = vgeo[i, j, k, _MJ, e]
                    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
                    z = vgeo[i,j,k,_z,e]

                    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
                    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

                    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

                    #          l_ρinv[i, j, k] = ρinv = 1 / ρ
                    l_ρinv[i, j, k] = ρinv = 1 / ρ
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
                end
            end
        end

        @synchronize

        @loop for k in (1:Nq; threadIdx().z)
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)
                    # TODO: Prefetch MJI and rhs

                    rhsU = rhsV = rhsW = rhsρ = rhsE = zero(DFloat)
                    MJI = vgeo[i, j, k, _MJI, e]

                    # buoyancy term
                    ρ = Q[i, j, k, _ρ, e]
                    rhsW -= ρ * gravity

                    # loop of ξ-grid lines
                    for n = 1:Nq
                        MJI_Dni = MJI * s_D[n, i]
                        MJI_Dnj = MJI * s_D[n, j]
                        MJI_Dnk = MJI * s_D[n, k]

                        rhsρ += MJI_Dni * s_F[n, j, k, _ρ]
                        rhsρ += MJI_Dnj * s_G[i, n, k, _ρ]
                        rhsρ += MJI_Dnk * s_H[i, j, n, _ρ]

                        rhsU += MJI_Dni * s_F[n, j, k, _U]
                        rhsU += MJI_Dnj * s_G[i, n, k, _U]
                        rhsU += MJI_Dnk * s_H[i, j, n, _U]

                        rhsV += MJI_Dni * s_F[n, j, k, _V]
                        rhsV += MJI_Dnj * s_G[i, n, k, _V]
                        rhsV += MJI_Dnk * s_H[i, j, n, _V]

                        rhsW += MJI_Dni * s_F[n, j, k, _W]
                        rhsW += MJI_Dnj * s_G[i, n, k, _W]
                        rhsW += MJI_Dnk * s_H[i, j, n, _W]

                        rhsE += MJI_Dni * s_F[n, j, k, _E]
                        rhsE += MJI_Dnj * s_G[i, n, k, _E]
                        rhsE += MJI_Dnk * s_H[i, j, n, _E]
                    end

                    rhs[i, j, k, _U, e] += rhsU
                    rhs[i, j, k, _V, e] += rhsV
                    rhs[i, j, k, _W, e] += rhsW
                    rhs[i, j, k, _ρ, e] += rhsρ
                    rhs[i, j, k, _E, e] += rhsE
                end
            end
        end

        # loop over moist variables
        # FIXME: Currently just passive advection
# TODO: This should probably be unrolled by some factor
rhsmoist = zero(eltype(rhs))
for m = 1:nmoist
    s = _nstate + m

    @synchronize

    @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                MJ = vgeo[i, j, k, _MJ, e]
                ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

                Qmoist = Q[i, j, k, s, e]
                U = Q[i, j, k, _U, e]
                V = Q[i, j, k, _V, e]
                W = Q[i, j, k, _W, e]

                fx = U * l_ρinv[i, j, k] * Qmoist
                fy = V * l_ρinv[i, j, k] * Qmoist
                fz = W * l_ρinv[i, j, k] * Qmoist

                s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
            end
        end
    end

    @synchronize

    @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                # TODO: Prefetch MJI and rhs
                MJI = vgeo[i, j, k, _MJI, e]

                rhsmoist = zero(DFloat)
                for n = 1:Nq
                    MJI_Dni = MJI * s_D[n, i]
                    MJI_Dnj = MJI * s_D[n, j]
                    MJI_Dnk = MJI * s_D[n, k]

                    rhsmoist += MJI_Dni * s_F[n, j, k, 1]
                    rhsmoist += MJI_Dnj * s_G[i, n, k, 1]
                    rhsmoist += MJI_Dnk * s_H[i, j, n, 1]
                end
                rhs[i, j, k, s, e] += rhsmoist
            end
        end
    end
end

# Loop over trace variables
# TODO: This should probably be unrolled by some factor
rhstrace = zero(eltype(rhs))
for m = 1:ntrace
    s = _nstate + nmoist + m

    @synchronize

    @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                MJ = vgeo[i, j, k, _MJ, e]
                ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

                Qtrace = Q[i, j, k, s, e]
                U = Q[i, j, k, _U, e]
                V = Q[i, j, k, _V, e]
                W = Q[i, j, k, _W, e]

                fx = U * l_ρinv[i, j, k] * Qtrace
                fy = V * l_ρinv[i, j, k] * Qtrace
                fz = W * l_ρinv[i, j, k] * Qtrace

                s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
            end
        end
    end

    @synchronize

    @loop for k in (1:Nq; threadIdx().z)
        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                # TODO: Prefetch MJI and rhs
                MJI = vgeo[i, j, k, _MJI, e]

                rhstrace = zero(DFloat)
                for n = 1:Nq
                    MJI_Dni = MJI * s_D[n, i]
                    MJI_Dnj = MJI * s_D[n, j]
                    MJI_Dnk = MJI * s_D[n, k]

                    rhstrace += MJI_Dni * s_F[n, j, k, 1]
                    rhstrace += MJI_Dnj * s_G[i, n, k, 1]
                    rhstrace += MJI_Dnk * s_H[i, j, n, 1]
                end
                rhs[i, j, k, s, e] += rhstrace
            end
        end
    end
end
end
nothing
end
# }}}

# {{{ volume_v3
function volumerhs_v3!(::Val{DEV},
                       ::Val{3},
                       ::Val{N},
                       ::Val{nmoist},
                       ::Val{ntrace},
                       rhs,
                       Q,
                       vgeo,
                       gravity,
                       D,
                       nelem) where {DEV, N, nmoist, ntrace}
  @setup DEV

  DFloat = eltype(Q)

  nvar = _nstate + nmoist + ntrace

  Nq = N + 1

  s_D = @shmem DFloat (Nq, Nq)
  s_F = @shmem DFloat (Nq, Nq, _nstate)
  s_G = @shmem DFloat (Nq, Nq, _nstate)
  s_H = @shmem DFloat (Nq, Nq, _nstate)

  r_rhsρ = @scratch DFloat (Nq, Nq, Nq) 2
  r_rhsU = @scratch DFloat (Nq, Nq, Nq) 2
  r_rhsV = @scratch DFloat (Nq, Nq, Nq) 2
  r_rhsW = @scratch DFloat (Nq, Nq, Nq) 2
  r_rhsE = @scratch DFloat (Nq, Nq, Nq) 2

  @inbounds @loop for e in (1:nelem; blockIdx().x)
    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)
        for k in (1:Nq)
          r_rhsρ[k, i, j] = 0
          r_rhsU[k, i, j] = 0
          r_rhsV[k, i, j] = 0
          r_rhsW[k, i, j] = 0
          r_rhsE[k, i, j] = 0
        end

        # fetch D into shared
        s_D[i, j] = D[i, j]
      end
    end

    for k in (1:Nq)
      @synchronize
      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)

          # Load values will need into registers
          MJ = vgeo[i, j, k, _MJ, e]
          ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
          ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
          ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
          z = vgeo[i,j,k,_z,e]

          U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
          ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

          P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

          #          l_ρinv[i, j, k] = ρinv = 1 / ρ
          # l_ρinv[i, j, k] 
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

            r_rhsρ[n,i,j] += Dkn * r_Hρ
            r_rhsU[n,i,j] += Dkn * r_HU
            r_rhsV[n,i,j] += Dkn * r_HV
            r_rhsW[n,i,j] += Dkn * r_HW
            r_rhsE[n,i,j] += Dkn * r_HE
          end
        end
      end

      @synchronize

      @loop for j in (1:Nq; threadIdx().y)
        @loop for i in (1:Nq; threadIdx().x)

          # loop of ξ-grid lines
          for n = 1:Nq
            Dni = s_D[n, i]
            Dnj = s_D[n, j]

            r_rhsρ[k,i,j] += Dni * s_F[n, j, _ρ]
            r_rhsρ[k,i,j] += Dnj * s_G[i, n, _ρ]

            r_rhsU[k,i,j] += Dni * s_F[n, j, _U]
            r_rhsU[k,i,j] += Dnj * s_G[i, n, _U]

            r_rhsV[k,i,j] += Dni * s_F[n, j, _V]
            r_rhsV[k,i,j] += Dnj * s_G[i, n, _V]

            r_rhsW[k,i,j] += Dni * s_F[n, j, _W]
            r_rhsW[k,i,j] += Dnj * s_G[i, n, _W]

            r_rhsE[k,i,j] += Dni * s_F[n, j, _E]
            r_rhsE[k,i,j] += Dnj * s_G[i, n, _E]
          end 
        end 
      end
    end # k

    @loop for j in (1:Nq; threadIdx().y)
      @loop for i in (1:Nq; threadIdx().x)

        for k in (1:Nq)
          MJI = vgeo[i, j, k, _MJI, e]
          ρ = Q[i, j, k, _ρ, e]

          rhs[i, j, k, _U, e] += MJI*r_rhsU[k, i, j]
          rhs[i, j, k, _V, e] += MJI*r_rhsV[k, i, j]
          rhs[i, j, k, _W, e] += MJI*r_rhsW[k, i, j] - ρ * gravity
          rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k, i, j]
          rhs[i, j, k, _E, e] += MJI*r_rhsE[k, i, j]
        end
      end
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
  D = rand(rnd, DFloat, Nq, Nq)

  rhs_v1 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs_v1!(Val(3), Val(N), Val(nmoist), Val(ntrace), rhs_v1, Q, vgeo,
                DFloat(grav), D, 1:nelem)
  @show norm_v1 = norm(rhs_v1)

  Q = reshape(Q, Nq, Nq, Nq, nvar, nelem)
  rhs_v1 = reshape(rhs_v1, Nq, Nq, Nq, nvar, nelem)
  vgeo = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

  rhs_v2 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs_v2!(Val(:CPU), Val(3), Val(N), Val(nmoist), Val(ntrace), rhs_v2, Q,
                vgeo, DFloat(grav), D, nelem)
  @show norm_v2 = norm(rhs_v2)
  @show norm_v1 - norm_v2

  rhs_v3 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
  volumerhs_v3!(Val(:CPU), Val(3), Val(N), Val(nmoist), Val(ntrace), rhs_v3, Q,
                vgeo, DFloat(grav), D, nelem)
  @show norm_v3 = norm(rhs_v3)
  @show (norm_v1 - norm_v3) / norm_v1

  if HAVE_CUDA
    rhs_v3 = zeros(DFloat, Nq, Nq, Nq, nvar, nelem)
    d_Q = CuArray(Q)
    d_D = CuArray(D)
    d_vgeo = CuArray(vgeo)
    d_rhs_v3 = CuArray(rhs_v3)

    @cuda(threads=(N+1, N+1), blocks=nelem,
          volumerhs_v3!(Val(:GPU), Val(3), Val(N), Val(nmoist), Val(ntrace),
                        d_rhs_v3, d_Q, d_vgeo, DFloat(grav), d_D, nelem))
    rhs_v3 .= d_rhs_v3
    @show norm_v3 = norm(rhs_v3)
    @show norm_v1 - norm_v3
  end

  nothing
end

if(false)
  for N = 1:7
    for Cp = 15:20
      Ndofs = 2^Cp
      Nel = Int(ceil(Ndofs/((N+1)*(N+1)*(N+1))))
      main(Nel, N, Float32)
    end
  end
end

main(4000,4,Float32)

# @device_code_ptx(main(4000,4,Float32)) 
