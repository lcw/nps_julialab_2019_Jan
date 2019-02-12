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

function knl!(::Val{DEV}, a) where {DEV}
  @setup DEV

  # r_a = @scratch Float32 (4, 5, 5) 2
  r_a = MArray{Tuple{4}, Float32}(undef)

  for k in 1:4
    r_a[k] = 0
  end

  nothing
end

function main()
  if HAVE_CUDA
    a = zeros(Float32, 4000)
    d_a = CuArray(a)

    @cuda(threads=(5, 5), blocks=4000, knl!(Val(:GPU), d_a))

    a .= d_a
  end

  nothing
end

main()
