# Intel LNL iGPU (Intel Xe2) — Quick Skill Notes

## Target Platform
- Device: Intel LNL iGPU, `Intel(R) Xe2 Graphics`
- GPU frequency: **2.0 GHz**
- Topology: **8 Xe**, **8 EU per Xe**

## CM (C for Metal) Execution Model
- A CM kernel launches as thread workgroups.
- One thread workgroup is scheduled onto one Xe.
- Typical workgroup sizes: **8**, **16**, **32** threads.
- Each thread is responsible for a specific slice of the kernel task.

## Practical Performance Envelope (Given)
- Peak memory bandwidth: **100 GB/s**
- Peak compute (FP16): **30 GFLOPS**

## Optimization Heuristics
1. Match workgroup size to kernel shape (`8/16/32`) and occupancy behavior.
2. Keep memory access contiguous/coalesced to approach bandwidth limits.
3. Reduce redundant loads/stores; reuse data in registers/shared paths where possible.
4. Balance compute and memory traffic; many kernels are bandwidth-limited before compute-limited.
5. Profile per-kernel timeline and compare CM vs OCL paths with repeatable conditions.

## Rule-of-Thumb Analysis
- Arithmetic intensity estimate:
	\[
		ext{AI}_{\text{balance}} = \frac{30\ \text{GFLOPS}}{100\ \text{GB/s}} = 0.3\ \text{FLOP/byte}
	\]
- Kernels below this intensity are likely memory-bound; above it, more likely compute-bound.

## Benchmarking Checklist
- Fix GPU frequency/power mode before runs.
- Use identical inputs, repeats, and warmup policy.
- Isolate each case into separate trace directories.
- Report min/avg/max and speedup against a clear baseline.
