# ⚙️ Universal AI-CAD Generator: Automated 2D Drawing & Image to 3D Parametric CAD Solid Modeling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Trimesh](https://img.shields.io/badge/Trimesh-5.1-brightgreen)](https://trimesh.org/)
[![ReportLab](https://img.shields.io/badge/ReportLab-5.0-D9381E)](https://www.reportlab.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Unit%20Tests-Passing%20(100%25)-brightgreen)](verify_aicad.py)

> **Final Year Engineering Capstone Project & IEEE Research Paper Prototype**  
> An open-source, deterministic computer vision and parametric CAD synthesis framework for mechanical engineering drawings and object photographs. Generates watertight 3D solids, automated Design for Manufacturability (DFM) audits, and industrial multi-format exports (STL, STEP AP214, OBJ, DXF, Excel BoQ, and ReportLab PDF dossiers).

---

## 🌟 1. System Overview & Key Capabilities

The **Universal AI-CAD Generator** bridges the gap between 2D engineering blueprints / real-world photographs and physical 3D manufacturing. Unlike unconstrained deep-learning generative models that output non-manifold polygonal soups with hallucinated dimensions, this system employs **deterministic computer vision heuristics** combined with **rigorous parameter-driven Constructive Solid Geometry (CSG)**.

### Supported Engineering Categories

| Category | Typical Input | Modeling Method | Key Output Features |
| :--- | :--- | :--- | :--- |
| **Gears & Power Transmission** | Blueprint / Drawing | Involute Gear Mathematical Solid | Configurable module ($m$), tooth count ($Z$), face width, bore diameter, and standard keyway slot. |
| **Stepped Transmission Shafts** | Blueprint / Orthographic | Multi-Stage Coaxial Cylinders | Main journal, bearing seats, drive couplings, step fillets, and keyways. |
| **Structural Brackets & Mounts** | 2D Technical Drawing | Extruded Polygonal Angle Flange | L-bracket arms, mounting bolt holes, and stiffening chamfered gusset. |
| **Electric Induction Motors** | Technical Drawing / Photo | Composite Cylindrical Assembly | Stator housing, cooling fins, drive shaft, end-shields, mounting feet, and terminal box. |
| **Automotive Vehicle Bodies** | Real-World Photograph | Aerodynamic Parametric Envelope | Wheelbase, length, width, height, ground clearance, and recessed wheel wells. |
| **Architectural Room Interiors** | Interior Room Photo | Spatial Bounding Wireframe Solid | 2D floorplan layout, wall height extrusion, furniture bounding volumes, and room code compliance. |

---

## 🔬 2. Technical Architecture & Engineering Modules

```
[2D Engineering Drawing / Photo]
               │
               ▼
┌────────────────────────────────────────────────────────┐
│ 1. Computer Vision Pipeline (vision_pipeline.py)       │
│  • Grayscale, Gaussian Blur (5x5, σ=1.2), Otsu & Canny │
│  • Suzuki-Abe Border Following Contour Topology        │
│  • Hough Circle Transform (Bore & Bolt Circle pitch)   │
│  • Drawing vs. Photograph Input Mode Classifier        │
│  • Frame Border & Title Block Rejection Filtering      │
│  • Geometric Classification (AR, Circularity, Solidity)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Dimension Confirmation & Interactive GUI (app.py)   │
│  • Transparent Candidate Parameters Display            │
│  • Human-in-the-Loop Dimension Editing (No Hallucinated│
│    or Invented Measurements)                           │
│  • Real-Time Interactive 3D Plotly Visualizer          │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Parametric Solid Modeling & DFM Engine (cad_engine) │
│  • Exact Involute & CSG Boolean Operations (Shapely)   │
│  • 2D Constrained Polygon Triangulation (Mapbox-Earcut)│
│  • Watertight Manifold 3D Mesh Solid (Trimesh)         │
│  • Material Mass & COG (7 Engineering Alloys/Polymers) │
│  • Automated DFM Feasibility Audit (Boothroyd Heuristic│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Multi-Format CAD Export & Certification Hub         │
│  • STL (Binary / ASCII) for 3D Printing & CAM Slicing  │
│  • Wavefront OBJ with Standard Normal Vectors          │
│  • ISO 10303-21 STEP (AP214 Faceted B-Rep Solid)       │
│  • AutoCAD DXF for 2D CNC Routing & Laser Cutting      │
│  • Costed Excel Bill of Quantities (XLSX BoQ)          │
│  • ReportLab 2-Page Engineering Certification Dossier  │
└────────────────────────────────────────────────────────┘
```

---

## 📐 3. Implemented vs. Template Features (Honest Research Disclosure)

In alignment with academic integrity guidelines for final year projects:

- **Fully Implemented Deterministic Algorithms:**
  - OpenCV image preprocessing, Canny edge detection, Otsu thresholding, and Suzuki-Abe contour extraction.
  - Automatic discrimination between Engineering Technical Drawings (including cyan blueprints) and Real-World Photographs.
  - Drawing frame border and title block rejection filtering.
  - Rigorous mathematical involute spur gear tooth profile generation with keyway bore subtraction.
  - Multi-stage coaxial cylinder synthesis for transmission shafts with keyway slots.
  - Structural L-bracket extrusion with central stiffening gusset and clearance mounting holes.
  - Assembly synthesis of electric induction motor stator, drive shaft, end-shields, and mounting feet.
  - Physical volume, surface area, center of gravity, and mass calculation across 7 real engineering materials (AISI 1045 steel, 6061-T6 aluminum, Class 30 cast iron, C36000 brass, Ti-6Al-4V titanium, ABS/PLA).
  - Automated Design for Manufacturability (DFM) scoring based on slenderness ratio, aspect ratio, and wall thickness.
  - All 6 export formats (STL, STEP AP214, OBJ, DXF, XLSX, ReportLab PDF).

- **Parametric Template Approximations (Clearly Disclosed):**
  - **Automotive Vehicle Bodies:** Monocular 2D photographs cannot capture occluded internal powertrain components (transmission, suspension, engine block). The generated vehicle is an **aerodynamic envelope solid** with user-confirmed dimensions $(L, W, H)$ and wheel wells.
  - **Electric Induction Motors:** Monocular drawings capture external dimensions (frame size, shaft length, feet). Internal stator laminations and copper coils are not rendered without internal assembly cross-sections.
  - **Text Dimension Parsing:** Scanned blueprint text varies greatly in font quality. While candidate geometric dimensions are inferred from contour aspect ratios, the user must **confirm or edit the dimensions** in the sidebar before solid synthesis.

---

## 💻 4. Step-by-Step Windows CMD Instructions

### Prerequisites
- Windows 10 or 11 (64-bit AMD64)
- Python 3.10 to 3.13 installed (ensure *"Add Python to PATH"* was checked during installation).

### Step 1: Open Windows Command Prompt
Press `Win + R`, type `cmd`, and press `Enter`.

### Step 2: Navigate to Project Directory
```cmd
cd /d "C:\Users\simbu T\.gemini\antigravity\scratch\ai-room-reconstruction"
```

### Step 3: Activate the Virtual Environment
```cmd
call .venv\Scripts\activate.bat
```
*(Your prompt will now show `(.venv)`).*

### Step 4: Run the Automated Test Suite
Verify that all vision, 3D modeling, DFM, and export pipelines function with 100% pass rate:
```cmd
python verify_aicad.py
```
**Expected Output:**
```
=================================================================
STARTING UNIVERSAL AI-CAD GENERATOR VALIDATION SUITE
=================================================================
[Spur Gear] -> Mode: Engineering Technical Drawing (82.0%) | Category: Spur / Helical Gear (94.0%)
[Stepped Shaft] -> Mode: Engineering Technical Drawing (82.0%) | Category: Stepped Transmission Shaft (92.5%)
[Electric Motor] -> Mode: Engineering Technical Drawing (82.0%) | Category: Electric Induction Motor (91.0%)
[L-Bracket] -> Mode: Engineering Technical Drawing (82.0%) | Category: Structural Angle Bracket (91.5%)
[Vehicle Body] -> Mode: Ordinary Real-World Photograph (95.5%) | Category: Automotive Vehicle Body (86.5%)
...
ALL UNIVERSAL AI-CAD GENERATOR TESTS COMPLETED SUCCESSFULLY!
=================================================================
```

### Step 5: Launch the Streamlit Web Application
```cmd
streamlit run app.py
```
Or simply double-click **`run.bat`** in the project directory!

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 🧪 5. Testing with Representative Sample Files

The repository includes high-resolution test blueprints and photographs in the `assets/` folder:

| Test Case | Asset File | How to Test in Web App |
| :--- | :--- | :--- |
| **1. Involute Spur Gear** | `assets/sample_gear_drawing.png` | Click *"Load Sample 1: Spur Gear Drawing"*, verify 94% classification, confirm tooth count $Z=20$, click *"Generate 3D Solid CAD Model"*. |
| **2. Stepped Shaft** | `assets/sample_shaft_drawing.png` | Click *"Load Sample 2: Stepped Shaft Drawing"*, confirm length $L=300\text{mm}$, generate 3D model, inspect step transitions. |
| **3. Electric Motor** | `assets/sample_motor_drawing.png` | Click *"Load Sample 3: Electric Motor Drawing"*, confirm Frame $\varnothing 180\text{mm}$, view stator housing and terminal box. |
| **4. Angle Bracket** | `assets/sample_bracket_drawing.png` | Click *"Load Sample 4: L-Bracket Drawing"*, view extruded flange with stiffening gusset and bolt holes. |
| **5. Vehicle Photograph** | `assets/sample_vehicle_photo.png` | Click *"Load Sample 5: Vehicle Photo"*, select *"Ordinary Real-World Photograph"*, generate aerodynamic envelope solid. |

---

## 📄 6. IEEE Research Documentation

A complete, publication-grade research manuscript is provided in:
- **`RESEARCH_PAPER_IEEE.md`** — Contains abstract, verified literature citations (Canny 1986, Suzuki & Abe 1985, ISO 10303-21, Boothroyd DFM), mathematical equations, system architecture diagram, experimental comparison table, and discussion of technical limitations.

---

## 📦 7. Project Structure

```
ai-room-reconstruction/
├── app.py                      # Main Streamlit web application & GUI
├── cad_engine.py               # Parametric 3D solid modeling & DFM engine
├── vision_pipeline.py          # OpenCV image processing & feature extraction
├── pdf_generator.py            # Publication-grade ReportLab PDF generator
├── verify_aicad.py             # Automated end-to-end verification test suite
├── RESEARCH_PAPER_IEEE.md      # IEEE conference-style research documentation
├── requirements.txt            # Locked pip dependencies
├── run.bat                     # Windows 1-click startup batch script
├── README.md                   # Complete project documentation
├── LICENSE                     # MIT Open Source License
└── assets/                     # Sample engineering drawings and photographs
    ├── sample_gear_drawing.png
    ├── sample_shaft_drawing.png
    ├── sample_motor_drawing.png
    ├── sample_bracket_drawing.png
    ├── sample_vehicle_photo.png
    ├── sample_bedroom.png
    └── sample_living_room.png
```

---

## 📜 8. License & Academic Attribution

This project is released under the **MIT License**. It was developed as a Final Year Capstone Project in Computer Science and Mechanical Systems Automation. If used for research or university evaluations, please cite `RESEARCH_PAPER_IEEE.md`.
