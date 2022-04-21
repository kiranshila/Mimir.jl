using SIGPROC, Plots, Dedisp, BenchmarkTools, CUDA, Clustering, DataFrames
using CUDA.CUFFT
pyplot()
default(; fmt=:png)

# Read Filterbank Data
fb = Filterbank("/home/kiran/Downloads/frb_data/injectfrb_nfrb10_DM100-500_21sec_20220420-1005.fil")

# Build plan
freqs = cu(collect(fb.data.dims[2]))
f_min, f_max = extrema(fb.data.dims[2])
n_samp = length(fb.data.dims[1])
#n_samp = 100000
δt = fb.headers["tsamp"]
t_total = n_samp * δt
dm_max = t_total / (Dedisp.KDM * (f_min^-2 - f_max^-2)) / 2
#dm_max = 1000
dm_min = 10
n_dm = 1024
dms = cu(collect(range(; start=dm_min, stop=dm_max, length=n_dm)))
plan = plan_dedisp(freqs, f_max, dms, δt)

# Preallocate output
n_chan = fb.headers["nchans"]
n_samp_out = n_samp ÷ 2
output = CUDA.zeros(n_samp_out, n_dm)

# Dedisperse
pulse = cu(fb.data.data[1:n_samp, :])
out = dedisp!(output, pulse, plan)

# Pretty Plot
# heatmap((δt .* (1:(n_samp ÷ 2))), Array(dms), Array(standardize(out))'; clims=(6, 15),xlabel="Starting Time Offset (s)", ylabel="DM", c=:jet)
# heatmap(Array(out))

# Find Peaks
function gaussian_kernel(fwhm, width=nothing; dtype=Float32)
    σ = fwhm / 2√(log(2))
    if isnothing(width)
        width = ceil(Int, 10 * σ)
    end
    raw_kern = [1 / (√(2π) * σ) * exp(-(i - 0.5 - width / 2)^2 / (2 * σ^2))
                for i in 1:width]
    return dtype.(raw_kern)
end

width_filter(dms, kernel, plan) = standardize(plan \ ((plan * dms) .* kernel))

search_widths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
width_kernels = reduce(hcat,
                       rfft(cu(fftshift(gaussian_kernel(width, n_samp_out))))
                       for width in search_widths)

function cluster_candidates(dm_transformed_data, width_kernels, starting_times, dms, widths;
                            snr_threshold=6,
                            radius=100, min_cluster_size=5)
    # Build FFT plan for width matched filtering
    width_search_fft_plan = plan_rfft(dm_transformed_data, 1)
    # Initialize peak index matrix
    peak_mat = Matrix{Integer}(undef, 0, 3)
    # Find peaks
    for (width_idx, kernel) in enumerate(eachcol(width_kernels))
        width_filtered_dm = width_filter(dm_transformed_data, kernel, width_search_fft_plan)
        # Do the search on the GPU and then bring back to CPU
        peaks = Array(findall(x -> x > snr_threshold, width_filtered_dm))
        peak_mat = [peak_mat; hcat(getindex.(peaks, [1 2]), fill(width_idx, length(peaks)))]
    end
    # Format the peaks matrix to hand over to dbscan
    peak_idxs = CartesianIndex.(eachcol(peak_mat[:,1:2])...)
    peak_mat = Float32.(peak_mat')
    # Cluster!
    clusters = dbscan(peak_mat, radius; min_cluster_size=min_cluster_size)
    # Initialize candidates data frame
    candidates = DataFrame(; snr=Float32[], start=Float32[], dm=Float32[], width=Float32[])
    # Fill DF
    for cluster in clusters
        # This is which cols from peak_mat are the indices in the cluster
        cluster_idxs = cluster.core_indices

        snrs = standardize(dm_transformed_data)[peak_idxs[cluster_idxs]]

        max_snr, local_idx = findmax(snrs)
        max_snr_idx = peak_idxs[cluster_idxs][local_idx]

        starting_time = starting_times[Tuple(max_snr_idx)[1]]
        dm = dms[Tuple(max_snr_idx)[2]]
        #width = widths[Tuple(max_snr_idx)[3]]

        push!(candidates, [max_snr, starting_time, dm, 0.0])
    end
    # Sort by SNR
    return sort(candidates, :snr; rev=true)
end