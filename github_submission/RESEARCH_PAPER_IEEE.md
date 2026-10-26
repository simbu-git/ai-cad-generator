# An Open-Source Parameter-Driven Computer Vision Framework for Automated 3D CAD Reconstruction and Manufacturing Feasibility Assessment

**Author:** Simbu T  
**Department:** Department of Computer Science & Engineering  
**Academic Year:** 2025 – 2026  
**Document Classification:** IEEE Technical Conference & Capstone Research Paper  

---

## Abstract
Traditional computer-aided design (CAD) reverse-engineering workflows require labor-intensive manual measurements, geometric parameterization, and solid modeling within proprietary CAD software suites. While deep-learning-based single-view 3D reconstruction networks have advanced for generalized mesh approximations, they often output non-manifold, unconstrained polygonal soups lacking dimensional intent, engineering tolerances, and standard ISO CAD interchange formats (STEP AP214, IGES, DXF). This paper presents the architecture, implementation, and empirical validation of **Universal AI-CAD Generator**, a modular, deterministic, parameter-driven framework that bridges monocular computer vision and mechanical solid modeling. The system integrates dual-mode classification (technical drawings vs. real-world photographs), contour topology extraction via Border Following algorithms, candidate feature parameterization, interactive human-in-the-loop dimension validation, manifold Constructive Solid Geometry (CSG) synthesis, automated Design for Manufacturability (DFM) heuristic auditing, and multi-format export (STL, STEP AP214, OBJ, DXF, XLSX Bill of Quantities, and ReportLab engineering dossiers). Empirical evaluation across five benchmark mechanical categories—involute spur gears, stepped transmission shafts, structural angle brackets, electric induction motors, and automotive bodies—demonstrates 100% manifold solid generation with strictly positive physical volumes and zero unconstrained hallucinations.

**Index Terms**—Computer-Aided Design (CAD), Reverse Engineering, Computer Vision, Contour Analysis, Involute Geometry, ISO 10303-21 STEP, Design for Manufacturability (DFM), Parametric Modeling.

---

## 1. Introduction & Problem Statement

### 1.1 Background
Engineering drawings and technical blueprints remain the primary medium for communicating geometric specifications, tolerances, and assembly relationships in manufacturing industries. However, millions of legacy mechanical parts exist solely as 2D orthographic paper prints, scanned raster drawings, or field photographs. Modernizing these assets into editable 3D CAD models (e.g., ISO 10303 STEP B-Rep solids) currently requires skilled mechanical designers spending hours recreating sketches, revolutions, and extrusions.

### 1.2 The Hallucination & Non-Manifold Problem in Modern AI-3D Models
Recent deep generative models (e.g., NeRFs, TripoSR, Point-E, Shap-E) generate visual 3D approximations from single images. While visually convincing in rendering viewports, these representations suffer from severe deficiencies when applied to mechanical engineering:
1. **Lack of Dimensional Intent:** Outputs are arbitrary scale meshes rather than dimensioned engineering components with defined millimeters and tolerances.
2. **Non-Manifold Topology:** Meshes frequently possess self-intersecting facets, degenerate zero-area faces, open boundaries, and inverted normal vectors, rendering them unsliceable for CNC toolpath generation or additive manufacturing.
3. **No CAD Feature Tree:** Designers cannot modify tooth counts, bore diameters, or keyway tolerances because the representation is an unsegmented triangle mesh rather than a boundary-representation (B-Rep) solid.
4. **Fabrication of Unknown Dimensions:** Many black-box generative systems "hallucinate" measurements, creating catastrophic risk in safety-critical machinery.

### 1.3 Project Objectives
To address these limitations, this research presents a deterministic, transparent, and reproducible CAD reconstruction system governed by four engineering design rules:
- **Rule 1 (Dual-Domain Input Processing):** Distinguish automatically between 2D orthographic blueprints/drawings and real-world photographs using background uniformity, edge density, and color space distribution.
- **Rule 2 (No Fabricated Dimensions):** Feature extraction provides nominal candidate baselines, while requiring human-in-the-loop confirmation before solid synthesis.
- **Rule 3 (Strict Manifold Solid Geometry):** All synthesized CAD models must be mathematically watertight, oriented, closed 2-manifold volumes ($V > 0$).
- **Rule 4 (Standardized Industrial Interchange):** Outputs must include standard manufacturing formats: ISO 10303-21 STEP AP214, stereolithography (STL), AutoCAD DXF, and engineering certification dossiers.

---

## 2. Related Work

The foundational methodologies utilized in this framework derive from verified computer vision and computational geometry literature:

1. **Edge Detection and Gradient Operators:** Canny [1] established optimal criteria for step edge detection (low error rate, localized response, single response), which forms the front-end filtering stage in our vision pipeline.
2. **Topological Contour Extraction:** Suzuki and Abe [2] developed the hierarchical border following algorithm for digitized binary images, which we employ to isolate outer part perimeters, inner through-holes, and keyway slots.
3. **Involute Gear Mathematics:** Buckingham [3] and ISO 53 standards formulate the kinematic conjugate action of spur gears, defined by involute curves unwrapped from a base circle $r_b = r_p \cos(\alpha)$, ensuring constant velocity ratio and uniform line of action.
4. **Boundary Representation and CAD Interchange:** ISO 10303-21 [4] defines the standardized clear text encoding for exchange of product data (STEP), establishing topological entities (VERTEX_POINT, EDGE_CURVE, ADVANCED_FACE, CLOSED_SHELL, MANIFOLD_SOLID_BREP).
5. **Design for Manufacturability (DFM):** Boothroyd, Dewhurst, and Knight [5] formulated quantitative guidelines evaluating machining access, tool aspect ratios, wall thicknesses, and raw stock utilization to prevent unmanufacturable geometries.
6. **Circle and Feature Extraction:** Duda and Hart [6] generalized the Hough Transform for detection of parametric primitives such as cylindrical bores, bolt hole circles, and shaft journals.

---

## 3. System Architecture & Methodology

The Universal AI-CAD architecture is organized into four decoupled, verifiable pipelines:

```
[Input Drawing or Photo]
          │
          ▼
┌───────────────────────────────────────────────┐
│ 1. Computer Vision & Feature Analysis Pipeline│
│  - Grayscale & Gaussian Filtering (5x5, σ=1.2)│
│  - Otsu Adaptive Thresholding & Canny Edges   │
│  - Border Following (Suzuki-Abe Topology)     │
│  - Hough Transform (Concentric Circles)       │
│  - Blueprint vs. Photograph Discrimination    │
│  - Outer Frame & Title Block Rejection Filter │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 2. Classification & Parameter Ingestion       │
│  - Aspect Ratio, Circularity & Solidity Audit │
│  - Convexity Defects & Tooth Count Heuristics │
│  - Human-in-the-Loop Dimension Confirmation   │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 3. Parametric Solid Modeling Engine           │
│  - Involute Gear Extrusion with Keyway Bore   │
│  - Multi-Stage Coaxial Cylinder Stepped Shaft │
│  - Stator Housing, Shaft, Flange & Feet Motor │
│  - L-Flange with Chamfered Stiffener Gusset   │
│  - Automotive Aerodynamic Enclosure Solid     │
│  - Watertightness & Manifold Mesh Verification│
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│ 4. DFM Auditing & Multi-Format CAD Export     │
│  - Wall Thickness & Slenderness Ratio Audit   │
│  - Mass & Center-of-Gravity (7 Alloys/Resins) │
│  - Binary STL (3D Printing / Slicing)         │
│  - Wavefront OBJ (3D Visualization)           │
│  - ISO 10303-21 AP214 STEP (Faceted B-Rep)    │
│  - AutoCAD DXF (2D Profile CNC Routing)       │
│  - Costed Excel BoQ (Raw Material & Tooling)  │
│  - ReportLab 2-Page Certification PDF Dossier │
└───────────────────────────────────────────────┘
```

### 3.1 Vision Pipeline & Drawing Discrimination
The input image $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ is converted to the HSV color space $\mathbf{I}_{\text{HSV}}$ and grayscale $\mathbf{I}_{\text{gray}}$. The input mode classifier computes:
1. Mean color saturation $\mu_S = \frac{1}{HW}\sum_{x,y} \mathbf{I}_{\text{HSV}}(x,y,1)$.
2. Mean Hue $\mu_H$ and standard deviation $\sigma_H$ to detect monochrome cyan/navy blueprints ($90 \le \mu_H \le 135, \sigma_H < 28$).
3. Boundary margin variance $\sigma_{\text{corner}}$ across four $40 \times 40$ px corner patches.
4. Edge density $\rho_E = \frac{1}{HW}\sum_{x,y} \mathbb{I}(\mathbf{I}_{\text{Canny}}(x,y) > 0)$.

A drawing is identified when:
$$\text{IsDrawing} = \left((\mu_S < 20 \lor (\mu_S < 45 \land \sigma_{\text{corner}} < 22)) \land \rho_E < 0.16\right) \lor (\text{IsBlueprint})$$

### 3.2 Drawing Frame and Title Block Filtering
Scanned engineering drawings invariably contain an outer border frame and a bottom-right title block containing project metadata. If unfiltered, the outer frame dominates contour area calculations. The pipeline applies a geometric filter:
$$\text{Reject}(C) = \begin{cases} 
\text{True} & \text{if } w_C > 0.88 W \land h_C > 0.88 H \\
\text{True} & \text{if } x_C > 0.45 W \land y_C > 0.55 H \land \frac{\max(w_C, h_C)}{\min(w_C, h_C)} > 3.0 \land A_C < 0.10 WH \\
\text{False} & \text{otherwise}
\end{cases}$$
The largest remaining candidate contour $C^*$ is analyzed for circularity $K = \frac{4\pi A_C}{P_C^2}$, bounding box aspect ratio $R = \frac{\max(w,h)}{\min(w,h)}$, and solidity $S = \frac{A_C}{A_{\text{ConvexHull}}}$.

### 3.3 Parametric Modeling Engines

#### 3.3.1 Involute Spur Gear Modeling
The spur gear profile is parameterized by module $m$, tooth count $Z$, face width $b$, bore diameter $d_b$, and keyway dimensions $(w_k, h_k)$. Pitch diameter $d_p = m \cdot Z$, addendum $h_a = 1.0 m$, and dedendum $h_f = 1.25 m$ yield outer diameter $d_o = d_p + 2 h_a$ and root diameter $d_r = d_p - 2 h_f$. The tooth flanks are synthesized by discrete involute angular discretization:
$$\theta_i = \frac{2\pi i}{Z}, \quad r(\phi) = \sqrt{r_b^2 + (r_b \phi)^2}, \quad \phi \in [0, \phi_{\max}]$$
The center bore and keyway slot are unified via CSG boolean union $\mathcal{B} = \text{Circle}(d_b/2) \cup \text{Box}(-w_k/2, 0, w_k/2, d_b/2 + h_k)$ and subtracted from the tooth polygon before prismatic linear extrusion along the $Z$-axis.

#### 3.3.2 Stepped Transmission Shaft Modeling
A stepped shaft is represented as an ordered sequence of $k$ coaxial cylindrical stages $\mathcal{S}_i = (L_i, D_i)$, where each stage is constructed as a regular 32-gon prism centered at origin and positioned sequentially:
$$Z_i^{\text{start}} = \sum_{j=1}^{i-1} L_j, \quad Z_i^{\text{end}} = \sum_{j=1}^i L_j$$
Keyway slots on drive ends are carved using boolean box difference.

#### 3.3.3 Structural Angle Bracket Modeling
The L-bracket comprises a vertical arm $(H_v, W, T)$, a horizontal base $(H_h, W, T)$, a central triangular stiffening gusset of thickness $T_g = T / 2$, and cylindrical mounting clearance holes created via CSG boolean subtraction.

#### 3.3.4 Electric Induction Motor Housing
The motor model synthesizes an IEC-standard cylindrical stator housing with cooling fins, front and rear drive end-shields (bearing caps), an extended drive shaft with keyway, a rectangular terminal conduit box mounted on top, and dual mounting foot flanges with bolt holes.

#### 3.3.5 Automotive Vehicle Envelope Solid
For vehicles from ordinary photos, the system extracts the baseline ground clearance, wheelbase, and greenhouse contour to synthesize a streamlined parametric aerodynamic envelope with recessed wheel arches. This is clearly designated as an **envelope solid template**, avoiding misleading claims of internal mechanical assembly reverse-engineering.

---

## 4. Experimental Setup & Empirical Results

The pipeline was implemented in Python 3.13 utilizing OpenCV (`opencv-python-headless 5.0.0`), NumPy 2.5, Trimesh 5.1, Shapely 2.2, Mapbox-Earcut 2.1, Plotly 7.1, and ReportLab 5.0 on a standard AMD64 architecture.

Five representative benchmark test cases were evaluated:
1. **Case 1:** Scanned Spur Gear technical drawing (`sample_gear_drawing.png`).
2. **Case 2:** Stepped Transmission Shaft blueprint (`sample_shaft_drawing.png`).
3. **Case 3:** Electric Induction Motor blueprint (`sample_motor_drawing.png`).
4. **Case 4:** Structural Angle Bracket blueprint (`sample_bracket_drawing.png`).
5. **Case 5:** Automotive Sedan Photograph (`sample_vehicle_photo.png`).

### 4.1 Quantitative Test Metrics
The automated test harness (`verify_aicad.py`) verified geometric properties, manifold integrity, and file generation.

| Test Case | Mode Classification | Object Classification | Solid Volume ($V$) | Surface Area ($A$) | Computed Mass (AISI 1045) | Watertight? | DFM Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Spur Gear** | Eng. Drawing (82.0%) | Spur Gear (94.0%) | $77.3\text{ cm}^3$ | $197.8\text{ cm}^2$ | $0.606\text{ kg}$ | **Yes** (True) | 100% (High) |
| **Stepped Shaft**| Eng. Drawing (82.0%) | Stepped Shaft (92.5%) | $383.5\text{ cm}^3$ | $490.0\text{ cm}^2$ | $3.011\text{ kg}$ | **Yes** (True) | 100% (High) |
| **Electric Motor**| Eng. Drawing (82.0%) | Induction Motor (91.0%) | $7,467.6\text{ cm}^3$ | $4,133.8\text{ cm}^2$ | $58.621\text{ kg}$ | **Yes** (True) | 100% (High) |
| **L-Bracket** | Eng. Drawing (82.0%) | Angle Bracket (91.5%) | $575.9\text{ cm}^3$ | $821.5\text{ cm}^2$ | $4.520\text{ kg}$ | **Yes** (True) | 100% (High) |
| **Vehicle Body** | Real Photo (95.5%) | Vehicle Body (86.5%) | $7,186,484.6\text{ cm}^3$| $471,726.6\text{ cm}^2$| $56,413.9\text{ kg}$ | **Yes** (True) | 100% (High) |

### 4.2 Multi-CAD Export Verification

All six export formats were generated and programmatically validated for header integrity and non-zero byte payloads:
1. **Binary STL:** 50,684 bytes — Validated with IEEE 754 little-endian binary facet descriptors; zero degenerate triangles.
2. **Wavefront OBJ:** 32,971 characters — Validated vertex (`v`) and facet (`f`) index continuity.
3. **ISO 10303-21 STEP (AP214):** 1,441 characters — Header and Data blocks conform to ISO 10303 clear text syntax with faceted B-Rep schema.
4. **AutoCAD DXF:** 430 characters — Valid ASCII 2D polyline cross-section for laser cutting and CNC profiling.
5. **Costed Excel BoQ (XLSX):** 7,042 bytes — Valid OOXML workbook containing material densities, rough stock allowance, tooling costs, and scrap estimates.
6. **ReportLab Engineering Dossier (PDF):** 462,508 bytes — 2-page publication-grade PDF featuring title header, 3-view imagery, dimension schedule, DFM audit report, and engineer sign-off blocks.

---

## 5. Technical Limitations & Discussion

Honest assessment of the system's operational boundaries is essential for engineering deployment:

1. **Occlusion in Monocular Imagery:** Single-view photographs and 2D blueprints cannot reveal hidden interior features (e.g., internal motor stator copper windings, blind tapped threads, internal gearbox oil passages) without multi-view orthographic projections (front, top, side cross-sections) or ultrasonic/CT scans.
2. **Vehicle Modeling Fidelity:** Vehicle models generated from photographs represent aerodynamic envelope solids based on wheelbase and silhouette contours. They are not structural reverse-engineered vehicle chassis and must not be used for finite element crash simulations without internal CAD sub-assembly modeling.
3. **Text / OCR Tolerancing:** While contour dimensions provide nominal geometric candidates, text OCR on noisy scanned blueprints can misread tolerance grades (e.g., mistaking `H7` for `H1` or `g6` for `66`). The system therefore enforces human-in-the-loop parameter confirmation to prevent dangerous machining errors.
4. **CAD Kernel Compatibility:** PythonOCC / CadQuery binary distributions currently lack precompiled binary wheels on Python 3.13 for Windows AMD64. The system reliably utilizes Trimesh with Shapely and Mapbox-Earcut for robust manifold solid generation and ISO 10303 faceted B-Rep schema generation.

---

## 6. Conclusion & Future Enhancements

The **Universal AI-CAD Generator** establishes a rigorous, open-source bridge between computer vision and manufacturing CAD generation. By eliminating unconstrained black-box hallucinations and replacing them with parameter-driven CSG synthesis, the system guarantees 100% manifold, watertight solids suitable for physical 3D printing, CNC machining, and engineering documentation.

Future work will focus on:
- Integration of multi-view orthographic drawing cross-referencing (Third Angle Projection reconciliation across Front, Top, and End elevations).
- Deep-learning-assisted engineering dimension text parsing using fine-tuned Donut/TrOCR models trained specifically on ISO 128 engineering typography.
- Integration with FreeCAD headless Python IPC for NURBS-based curved surface STEP representations.

---

## References

[1] J. Canny, "A Computational Approach to Edge Detection," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-8, no. 6, pp. 679–698, Nov. 1986. DOI: 10.1109/TPAMI.1986.4767851.  
[2] S. Suzuki and K. Abe, "Topological structural analysis of digitized binary images by border following," *Computer Vision, Graphics, and Image Processing*, vol. 30, no. 1, pp. 32–46, 1985. DOI: 10.1016/0734-189X(85)90016-7.  
[3] E. Buckingham, *Analytical Mechanics of Gears*, New York: McGraw-Hill, 1949; Dover Publications, 1988.  
[4] ISO 10303-21:2016, *Industrial automation systems and integration — Product data representation and exchange — Part 21: Implementation methods: Clear text encoding of the exchange structure*, International Organization for Standardization, Geneva, Switzerland, 2016.  
[5] G. Boothroyd, P. Dewhurst, and W. Knight, *Product Design for Manufacture and Assembly*, 3rd ed., Boca Raton, FL: CRC Press, 2010.  
[6] R. O. Duda and P. E. Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures," *Communications of the ACM*, vol. 15, no. 1, pp. 11–15, Jan. 1972. DOI: 10.1145/361237.361242.  
[7] D. H. Ballard, "Generalizing the Hough transform to detect arbitrary shapes," *Pattern Recognition*, vol. 13, no. 2, pp. 111–122, 1981. DOI: 10.1016/0031-3203(81)90009-1.  
[8] W. E. Lorensen and H. E. Cline, "Marching cubes: A high resolution 3D surface construction algorithm," *ACM SIGGRAPH Computer Graphics*, vol. 21, no. 4, pp. 163–169, 1987. DOI: 10.1145/37402.37422.  
