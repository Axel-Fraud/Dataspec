# Dataspec
Open-source Bayesian data analysis software with MCMC sampling, nonlinear regression, Poisson modeling, and interactive visualizations. Built in PyQt5 for scientists and students seeking intuitive tools for probabilistic inference and uncertainty quantification.

# DataSpec – Bayesian Data Analysis Software

**DataSpec** is a cross-platform desktop application for Bayesian data analysis, model fitting, and uncertainty quantification. It provides researchers and students with an intuitive graphical interface for performing probabilistic inference, nonlinear regression, and statistical modeling — all without needing to write a single line of code.

Originally developed to support scientific research in nuclear spectroscopy, DataSpec is designed for broader use across physics, environmental science, and data-driven research fields where transparency, uncertainty, and interpretability matter.

---

## 🔍 Key Features

- **Bayesian Curve Fitting**  
  Supports Gaussian, Voigt, and user-defined nonlinear models with full posterior parameter estimation.

- **MCMC Sampling with PyMC**  
  Leverages PyMC3/4 for probabilistic modeling, including NUTS and Metropolis-Hastings samplers, trace diagnostics, and posterior predictive checks.

- **Poisson Regression**  
  Tailored for low-count experimental data, especially in particle physics, nuclear data, and biomedical applications.

- **Drag-and-Drop Interface**  
  Easily load CSV files via GUI; preview, clean, and prepare datasets before modeling.

- **Uncertainty Visualization**  
  Automatically generates best-fit curves with shaded confidence regions, residuals, and parameter summaries.

- **Interactive Data Cleaning**  
  Rename columns, interpolate missing values, remove outliers — all within the interface.

- **Session History + Undo**  
  Every analysis step is tracked. Undo mistakes or revisit earlier configurations with a single click.

- **Modular Plugin Architecture**  
  Easily extend functionality — add your own models, prior structures, or data loaders.

---

## ⚡ Quick Start

1. Download the `.exe` or `.app` from the [Releases](https://github.com/yourusername/dataspec/releases) page  
2. Launch the app  
3. Drag in a CSV file  
4. Select a model, fit, and visualize the results

---

## 🧠 Supported Models

- Gaussian  
- Voigt (Gaussian-Lorentz convolution)  
- Poisson (ideal for count data)  
- Custom (define your own via expression input or plugin)

---

## 💻 Installation & Usage

### 🟢 Option 1: Use Prebuilt Application

1. Visit the [Releases](https://github.com/yourusername/dataspec/releases) page  
2. Download the latest version for your OS  
3. Run the executable — no setup required

### 🧪 Option 2: Run From Source

```bash
git clone https://github.com/yourusername/dataspec.git
cd dataspec
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
