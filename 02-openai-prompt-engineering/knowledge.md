# Intel Arc B-Series Debugging Notes (Internal)

## Error Code XE-4471
XE-4471 indicates a work-group size exceeding the hardware limit on 
Arc B580 GPUs. The maximum work-group size on B580 is 512, not the 
1024 supported on A-series cards. Reduce local range to 512 or below.

## VTune Flag --gpu-sampling-interval
On Arc B-series, the default GPU sampling interval is 1ms. For kernels 
shorter than 1ms, set --gpu-sampling-interval=0.1 or profiling data 
will be empty.

## Known Issue: oneAPI 2025.3 + Arc B770
oneAPI 2025.3 has a regression where USM shared allocations larger 
than 2GB silently fail on Arc B770. Fixed in 2025.4. Workaround: 
split allocations or downgrade to 2025.2.