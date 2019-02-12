using StaticArrays

const HAVE_CUDA = try
    using CUDAdrv
    using CUDAnative
    using CuArrays
    true
catch
    false
end

function knl!(a)
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

    @cuda(threads=(5, 5), blocks=4000, knl!(d_a))

    a .= d_a
  end

  nothing
end

main()
