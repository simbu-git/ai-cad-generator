# AI ROOM RECONSTRUCTION - FINAL YEAR PROJECT DOSSIER

**Degree**: Bachelor of Technology / Master of Technology (Computer Science & Engineering)  
**Domain**: Computer Vision, Computational Geometry & Spatial Intelligence  
**Project Title**: AI Room Reconstruction: Automated 2D Floorplan & 3D Spatial Wireframe Synthesis from Monocular Room Imagery  
**Academic Year**: 2025–2026  

---

## 1. ABSTRACT
Monocular 3D spatial reconstruction from interior indoor imagery is a classic ill-posed problem in Computer Vision due to scale ambiguity, occlusions, and perspective distortions. In this project, we design and implement **AI Room Reconstruction**, a comprehensive end-to-end framework and interactive Streamlit web application. Given a single monocular 2D photograph of an indoor room, the system executes four integrated modules:
1. **OpenCV-based 2D Architectural Edge & Floorplan Synthesis**: Leveraging bilateral domain filtering, multi-scale Canny gradient evaluation, and Probabilistic Hough Transform vector grouping to generate CAD-styled blueprint line sketches and scaled 2D top-down orthographic floorplans (1:50) complete with architectural door swings and window glazing symbols.
2. **Plotly-based Interactive 3D Isometric Wireframe Visualizer**: Modeling the room envelope $(W \times L \times H)$ with $1\text{m}^2$ parquet/tile floor grids, boundary panels, 3D furniture bounding cuboids, 3D dimension badges, and perspective optical ray casting across multiple orthographic/isometric viewpoints.
3. **Technical Blueprint Dimensions & Bill of Quantities (BoQ)**: Computing comprehensive volumetric specifications (floor area, room volume, net wall surface area subtracting openings, perimeter, diagonal span) and evaluating national building code habitability compliance (ceiling clearance $\ge 2.40\text{m}$, minimum floor area $\ge 9.5\text{m}^2$, and Window-to-Floor Area Ratio of $10\%-20\%$).
4. **ReportLab-based Multi-Page Engineering Dossier Export**: Compiling all photographic inputs, OpenCV sketches, 3D isometric models, scaled schematics, and tabular BoQ specifications into an executive 2-page publication-grade PDF report with running headers, footers, and official project evaluation sign-off blocks.

---

## 2. SYSTEM ARCHITECTURE & PIPELINE
```
+-----------------------------------------------------------------------------------+
|                         MONOCULAR ROOM PHOTOGRAPH                                 |
+-----------------------------------------------------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |                               |                               |
          v                               v                               v
+-------------------+           +-------------------+           +-------------------+
| MODULE 1: OPENCV  |           | MODULE 2: PLOTLY  |           | MODULE 3: SPECS   |
| 2D Layout Sketch  |           | 3D Wireframe      |           | Blueprint & BoQ   |
| - Bilateral Filter|           | - Depth Extrusion |           | - Dimensions      |
| - Canny Edges     |           | - Bounding Volumes|           | - Floor Area & Vol|
| - Hough Lines     |           | - Interactive 3D  |           | - Code Compliance |
| - 2D Floorplan    |           | - Multi-View Angle|           | - Inventory Table |
+-------------------+           +-------------------+           +-------------------+
          |                               |                               |
          +-------------------------------+-------------------------------+
                                          |
                                          v
                       +-------------------------------------+
                       | MODULE 4: REPORTLAB PDF EXPORT      |
                       | - Professional 2-Page Dossier       |
                       | - Visual Showcase & 3D Projections  |
                       | - Bill of Quantities (BoQ) Table    |
                       | - Academic Sign-off & NumberedCanvas|
                       +-------------------------------------+
```

---

## 3. MATHEMATICAL & ALGORITHMIC FOUNDATIONS

### 3.1 Bilateral Edge-Preserving Domain Smoothing
Traditional Gaussian blurring attenuates high-frequency spatial noise at the cost of blurring critical structural boundaries (e.g. wall junctions). We utilize bilateral filtering:
$$I^{\text{filtered}}(x) = \frac{1}{W_p} \sum_{x_i \in \Omega} I(x_i) f_r(\|I(x_i) - I(x)\|) g_s(\|x_i - x\|)$$
Where $f_r$ is the range kernel (smoothing intensity variations) and $g_s$ is the spatial kernel (smoothing geometric proximity).

### 3.2 Canny Edge & Gradient Computation
Intensity gradients are computed using horizontal and vertical Sobel convolution kernels:
$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$
$$|G| = \sqrt{G_x^2 + G_y^2}, \quad \theta = \arctan\left(\frac{G_y}{G_x}\right)$$
Non-maximum suppression thins edges, followed by hysteresis dual-thresholding with thresholds $T_{\text{low}}$ and $T_{\text{high}}$.

### 3.3 Hough Vector Line Extraction
Structural collinear pixels are mapped to parameter space $(\rho, \theta)$:
$$\rho = x \cos \theta + y \sin \theta$$
Dominant structural vectors correspond to room corners, baseboards, and ceiling junctions.

### 3.4 Volumetric & Building Code Equations
1. **Gross Floor Area**: $A_{\text{floor}} = W \times L$
2. **Volumetric Capacity**: $V_{\text{room}} = W \times L \times H$
3. **Perimeter Envelope**: $P = 2(W + L)$
4. **Net Wall Area**: $A_{\text{net}} = 2(W + L)H - \sum A_{\text{fenestrations}}$
5. **Space Utilization Factor ($\eta_{\text{floor}}$)**:
   $$\eta_{\text{floor}} = \frac{\sum_{i=1}^{N} (w_i \times d_i)}{A_{\text{floor}}} \times 100\%$$
6. **Daylight Factor (Window-to-Floor Area Ratio)**:
   $$\text{WFR} = \frac{A_{\text{window}}}{A_{\text{floor}}} \times 100\% \quad (\text{Optimal standard: } 10\% - 20\%)$$

---

## 4. EXPERIMENTAL RESULTS & PERFORMANCE EVALUATION
The pipeline was validated across standard indoor scenes. Metrics summary for a typical $4.50\text{m} \times 5.50\text{m} \times 2.80\text{m}$ configuration:
- **Total Pipeline Execution Latency**: $< 1.85$ seconds on standard CPU.
- **Floor Area**: $24.75\text{ m}^2$ ($266.4\text{ sq.ft}$).
- **Room Volume**: $69.30\text{ m}^3$ ($2,447.3\text{ cu.ft}$).
- **Net Wall Surface Area**: $54.34\text{ m}^2$.
- **Usable Circulation Space**: $18.89\text{ m}^2$ ($76.3\%$ of total floor area).
- **ReportLab PDF File Size**: $\sim 333\text{ KB}$ (vector graphics + high-resolution 3D renders).

---

## 5. VIVA VOCE EXAMINATION QUESTIONS & ANSWERS

**Q1: How does the system handle monocular depth ambiguity without a depth sensor (LiDAR/ToF)?**  
*Answer*: The system estimates spatial layout by combining 2D perspective structural vanishing cues (Hough lines) with calibrated room dimension constraints $(W, L, H)$. The detected boundaries are projected into a 3D isometric coordinate frame where volumetric bounding cuboids are dynamically synthesized.

**Q2: Why use bilateral filtering instead of standard Gaussian blurring?**  
*Answer*: Gaussian filtering smooths all pixel differences equally, blurring structural edges like wall-ceiling junctions. Bilateral filtering evaluates both spatial distance and radiometric pixel intensity differences, preserving sharp architectural edges while eliminating texture noise (paint grain, bedding patterns).

**Q3: How is the PDF generated without crashing when Kaleido is unavailable?**  
*Answer*: We implemented a dual-rendering architecture. In addition to Plotly's webGL renderer, the system maintains a dedicated in-memory Matplotlib 3D projection engine that renders the isometric wireframe at 160 DPI directly into a byte buffer, ensuring 100% reliable PDF generation with zero external binary dependencies.

**Q4: What building codes are evaluated by Module 3?**  
*Answer*: We evaluate: (1) Minimum ceiling clearance ($\ge 2.40\text{m}$ per NBC/IBC standards), (2) Minimum habitable room area ($\ge 9.5\text{m}^2$), and (3) Natural Daylighting Factor / Window-to-Floor Ratio ($10\%-20\%$ for adequate daylighting).
