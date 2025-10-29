# README.md

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/your_username/unified-fractal-novelty-framework">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Unified Fractal Novelty AI Framework</h3>

  <p align="center">
    A simulation framework integrating fractal dimensions, novelty dynamics, and physics-inspired models for domains like turbulence, brain activity, and galaxies.
    <br />
    <a href="https://github.com/your_username/unified-fractal-novelty-framework"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/your_username/unified-fractal-novelty-framework">View Demo</a>
    ·
    <a href="https://github.com/your_username/unified-fractal-novelty-framework/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/your_username/unified-fractal-novelty-framework/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This repository contains the Unified Fractal Novelty AI Framework, a Python-based simulation tool that models complex systems using fractal dimensions, novelty operators, and physics simulations. It supports domain-specific modeling for turbulence, brain dynamics, and galactic structures.

Key Features:
- Fractal dimension calculation via box-counting.
- Dynamic F-operator for novelty and integration.
- Matter, star, and black hole (Kerr metric) simulators.
- Validation suite with falsifiability tests.
- Domain-specific initializations using fractional Brownian motion (fBm) for realistic fractal structures.

Latest Version: v27 (KERR ACCURACY & BUG FIXES) - October 27, 2025

Changes in v27:
- Refined Kerr metric: Accurate effective potential for null geodesics, added r bounds to avoid singularities.
- Fixed initialization bugs: Ensured classes take parameters correctly with defaults.
- Added Kerr plotting: Polar r vs phi ('kerr_orbit.png').
- Complete code without truncations.

Previous Version: v21 (DOMAIN-SPECIFIC INITIALIZATIONS) - October 27, 2025

Changes in v21:
- Added domain-specific initial point clouds using fBm for target fractal dimensions (e.g., turbulence: D~2.36, brain: D~1.65, galaxy: D~1.3).
- Updated parameters with data_type and initial_fd_targets.
- Generalized simulators for domain matching.

Author: J. Asher (Enhanced by AI Assistant)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [![Python][Python]][Python-url]
* [![NumPy][NumPy]][NumPy-url]
* [![SciPy][SciPy]][SciPy-url]
* [![NetworkX][NetworkX]][NetworkX-url]
* [![Matplotlib][Matplotlib]][Matplotlib-url]
* [![Pandas][Pandas]][Pandas-url]
* [![Seaborn][Seaborn]][Seaborn-url]
* [![Scikit-learn][Scikit-learn]][Scikit-learn-url]
* Optional: Numba (for GPU/CPU acceleration)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To set up and run the framework locally, follow these steps.

### Prerequisites

- Python 3.12.3 or higher
- Install required libraries:
  ```sh
  pip install numpy scipy networkx matplotlib pandas seaborn scikit-learn numba
  ```

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/your_username/unified-fractal-novelty-framework.git
   ```
2. Navigate to the project directory:
   ```sh
   cd unified-fractal-novelty-framework
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

Run the simulation using the CLI:

```sh
python USimNewV3.py --data-type turbulence --steps 100 --nodes 200 --validate --real-data-file path/to/real_data.csv
```

### Arguments
- `--steps`: Number of simulation steps (default: 100).
- `--nodes`: Number of nodes in simulators (default: 200).
- `--params-file`: Path to JSON parameters file.
- `--validate`: Run validation suite.
- `--real-data-file`: Path to real data CSV for validation.
- `--data-type`: Domain type ('turbulence', 'brain', 'galaxy') (default: 'turbulence').
- `--monte-carlo`: Number of Monte Carlo trials (default: 1).
- `--gpu`: Enable GPU if available (requires Numba).

Outputs:
- `simulation_results.json`: Simulation data.
- `validation_report.json`: Validation results (if `--validate`).
- Plots: e.g., 'kerr_orbit.png', '3d_plot.png', animations like 'star_4d.gif'.

_For more examples, refer to the code's CLI epilog or run `python USimNewV3.py --help`._

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Kerr metric refinements.
- [x] Domain-specific fBm initializations.
- [ ] Add neural dynamics module for brain domain.
- [ ] Enhance GPU support for larger simulations.
- [ ] Integrate real-time visualization.

See the [open issues](https://github.com/your_username/unified-fractal-novelty-framework/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome! To contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

J. Asher - [LinkedIn](https://linkedin.com/in/your_username) - your_email@example.com

Project Link: [https://github.com/your_username/unified-fractal-novelty-framework](https://github.com/your_username/unified-fractal-novelty-framework)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [xAI](https://x.ai) for inspiration.
* Open-source libraries: NumPy, SciPy, NetworkX, Matplotlib.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/your_username/unified-fractal-novelty-framework.svg?style=for-the-badge
[contributors-url]: https://github.com/your_username/unified-fractal-novelty-framework/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/your_username/unified-fractal-novelty-framework.svg?style=for-the-badge
[forks-url]: https://github.com/your_username/unified-fractal-novelty-framework/network/members
[stars-shield]: https://img.shields.io/github/stars/your_username/unified-fractal-novelty-framework.svg?style=for-the-badge
[stars-url]: https://github.com/your_username/unified-fractal-novelty-framework/stargazers
[issues-shield]: https://img.shields.io/github/issues/your_username/unified-fractal-novelty-framework.svg?style=for-the-badge
[issues-url]: https://github.com/your_username/unified-fractal-novelty-framework/issues
[license-shield]: https://img.shields.io/github/license/your_username/unified-fractal-novelty-framework.svg?style=for-the-badge
[license-url]: https://github.com/your_username/unified-fractal-novelty-framework/blob/master/LICENSE
[product-screenshot]: images/screenshot.png
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[SciPy]: https://img.shields.io/badge/SciPy-%23013243.svg?style=for-the-badge&logo=scipy&logoColor=white
[SciPy-url]: https://scipy.org/
[NetworkX]: https://img.shields.io/badge/NetworkX-%23013243.svg?style=for-the-badge&logo=networkx&logoColor=white
[NetworkX-url]: https://networkx.org/
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23013243.svg?style=for-the-badge&logo=matplotlib&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/
[Seaborn]: https://img.shields.io/badge/Seaborn-%23013243.svg?style=for-the-badge&logo=seaborn&logoColor=white
[Seaborn-url]: https://seaborn.pydata.org/
[Scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit-learn-url]: https://scikit-learn.org/

# LICENSE

MIT License

Copyright (c) 2025 J. Asher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
