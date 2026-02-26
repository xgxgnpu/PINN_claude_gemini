# PINN High-Frequency Heat Equation: Gemini 3.1 vs Claude Sonnet 4.6

A head-to-head benchmark of two frontier LLMs — **Google Gemini 3.1 Pro** (released 2026-02-19) and **Anthropic Claude Sonnet 4.6** (released 2026-02-17) — on generating Physics-Informed Neural Network (PINN) code for solving a high-frequency heat equation forward problem. Both models received the **exact same single prompt** and produced complete, runnable Python scripts without any human modification.

## Problem Definition

| Item | Description |
|------|-------------|
| **PDE** | $u_t - \alpha\, u_{xx} = 0$ (heat equation) |
| **Diffusivity** | $\alpha = 0.01$ |
| **Spatial frequency** | $k = 5$ |
| **Exact solution** | $u(x,t) = \sin(k\pi x)\,\exp\!\bigl(-\alpha(k\pi)^2 t\bigr)$ |
| **Training strategy** | Two-phase: Adam (20,000 iters) + L-BFGS (up to 10,000 iters) |

## Benchmark Results

| Phase | Claude Sonnet 4.6 | Gemini 3.1 Pro |
|-------|-------------------|----------------|
| Adam (20k iters) | 308 s, L2 = 1.17e-3 | ~220 s, L2 = 2.24e-3 |
| L-BFGS | 1795 s, L2 = 4.83e-4 | ~20 s, L2 = 3.93e-4 |
| **Total time** | **2105 s (35 min)** | **~239 s (4 min)** |
| **Final L2 relative error** | **4.83 x 10⁻⁴** | **3.93 x 10⁻⁴** |

> Training speed differs by ~9x; Gemini achieves slightly better accuracy in a fraction of the time.

## Key Algorithmic Differences

| Aspect | Claude Sonnet 4.6 | Gemini 3.1 Pro |
|--------|-------------------|----------------|
| **Architecture** | Standard MLP (6 x 64, Tanh) | Fourier Feature Mapping + MLP (5 x 128, Tanh) |
| **Loss weighting** | Equal weights on PDE / IC / BC | 20x weight on IC and BC losses |
| **L-BFGS data** | Random resampling each step (breaks 2nd-order convergence) | Fixed collocation points (preserves convergence) |
| **Domain** | [0, 1] x [0, 1] | [-1, 1] x [0, 1] (harder: 2x spatial oscillations) |
| **Code style** | 534 lines, highly modular, detailed outputs | 308 lines, concise, research-oriented |

### Why does Gemini win on efficiency?

1. **Fourier Feature Mapping** overcomes the spectral bias of standard MLPs for high-frequency functions (Tancik et al., NeurIPS 2020).
2. **Weighted loss** balances the multi-task optimization inherent in PINNs (PDE residual vs. IC/BC constraints).
3. **Fixed training data** for L-BFGS preserves the deterministic objective assumption required by quasi-Newton methods, enabling true second-order convergence.

## Repository Structure

```
PINN_claude_gemini/
├── README.md
└── test1_PINN_forward/
    ├── run_claude_sonnet46.py        # Claude Sonnet 4.6 generated code
    ├── run_gemini31_pro.py           # Gemini 3.1 Pro generated code
    ├── output/
    │   ├── claude_sonnet46/          # Claude's training outputs
    │   │   ├── loss_history.txt
    │   │   ├── loss_history_phases.txt
    │   │   ├── prediction.txt
    │   │   ├── exact_solution.txt
    │   │   ├── error_map.txt
    │   │   ├── model.pth
    │   │   ├── summary.txt
    │   │   ├── loss_curves.png
    │   │   ├── solution_slices.png
    │   │   ├── solution_2d.png
    │   │   ├── solution_3d.png
    │   │   └── l2_convergence.png
    │   └── gemini31_pro/             # Gemini's training outputs
    │       ├── loss_and_l2_history.txt
    │       ├── exact_solution.txt
    │       ├── pred_solution.txt
    │       ├── error_absolute.txt
    │       ├── loss_and_l2_curve.png
    │       ├── solution_comparison.png
    │       └── time_slice_t05.png
    └── report/
        └── 01_Gemini31_vs_ClaudeS46_PINN_HighFreq_PDE_Benchmark.md
```

## Quick Start

### Requirements

- Python 3.8+
- PyTorch (with CUDA recommended)
- NumPy
- Matplotlib

### Run

```bash
cd test1_PINN_forward

# Run Claude Sonnet 4.6's implementation
python run_claude_sonnet46.py

# Run Gemini 3.1 Pro's implementation
python run_gemini31_pro.py
```

Both scripts auto-detect GPU and save all outputs (loss histories, solution data, plots) to `output/claude_sonnet46/` and `output/gemini31_pro/` respectively.

## Fairness Statement

Both generated scripts were verified to be free of any form of cheating:

- **No exact-solution leakage**: The analytical solution is only used post-training for L2 error evaluation, never during gradient computation.
- **Equivalent L2 metrics**: Both use relative L2 error; the formulas are mathematically equivalent despite surface differences.
- **Gemini solves a harder problem**: Its domain [-1, 1] contains twice the spatial oscillations compared to Claude's [0, 1], yet still achieves better accuracy faster.

See the [full benchmark report](test1_PINN_forward/report/01_Gemini31_vs_ClaudeS46_PINN_HighFreq_PDE_Benchmark.md) for detailed analysis.

## The Prompt

Both models received the identical prompt (translated):

> Provide a complete PINNs code for solving the high-frequency heat equation forward problem. Use the analytical solution only for computing the L2 test relative error. Print all losses, save data to txt files, and generate plots.
> 1. Use PyTorch with GPU training: Adam 20,000 iterations, L-BFGS 10,000 iterations
> 2. Accuracy requirement: at least 1e-2
> 3. Return a .py file

## Test Environment

- NVIDIA GPU + PyTorch
- Identical prompt for both models, single-shot generation, no human edits
- Gemini 3.1 Pro (released 2026-02-19) | Claude Sonnet 4.6 (released 2026-02-17)

## Author's Related Publications

The following publications by the repository author are related to neural network methods for solving PDEs:

1. **X. Xiong**, K. Lu, Z. Zhang, Z. Zeng, S. Zhou, R. Hu, Z. Deng, "[High-frequency flow field super-resolution via physics-informed hierarchical adaptive Fourier feature networks](https://pubs.aip.org/aip/pof/article/37/9/097111/3361448)," *Physics of Fluids*, vol. 37, no. 9, p. 097111, 2025.

2. **X. Xiong**, Z. Zhang, R. Hu, C. Gao, Z. Deng, "[Separated-variable spectral neural networks: a physics-informed learning approach for high-frequency PDEs](https://arxiv.org/abs/2508.00628)," *arXiv preprint arXiv:2508.00628*, 2025.

3. **X. Xiong**, K. Lu, Z. Zhang, Z. Zeng, S. Zhou, Z. Deng, R. Hu, "[J-PIKAN: A physics-informed KAN network based on Jacobi orthogonal polynomials for solving fluid dynamics](https://www.sciencedirect.com/science/article/pii/S1007570425008238)," *Communications in Nonlinear Science and Numerical Simulation*, p. 109414, 2025.

4. Z. Zhang, **X. Xiong**, S. Zhang, W. Wang, X. Yang, S. Zhang, C. Yang, "[A pseudo-time stepping and parameterized physics-informed neural network framework for Navier–Stokes equations](https://pubs.aip.org/aip/pof/article/37/3/033612/3338823)," *Physics of Fluids*, vol. 37, no. 3, p. 033612, 2025.

5. Z. Zhang, **X. Xiong**, S. Zhang, W. Wang, Y. Zhong, C. Yang, X. Yang, "[Legend-KINN: A Legendre Polynomial-Based Kolmogorov-Arnold-Informed Neural Network for Efficient PDE Solving](https://www.sciencedirect.com/science/article/pii/S0957417425034542)," *Expert Systems with Applications*, p. 129839, 2025.

6. Z. Zhang, **X. Xiong**, S. Zhang, Y. Zhao, X. Yang, "[Physics-Informed Neural Networks and Neural Operators for Parametric PDEs: A Human-AI Collaborative Analysis](https://arxiv.org/abs/2511.04576)," *arXiv preprint arXiv:2511.04576*, 2025.

## License

This project is released for educational and research purposes. The AI-generated code is provided as-is for benchmarking purposes.
