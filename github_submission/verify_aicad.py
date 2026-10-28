"""
Automated unit test suite verifying all modules of Universal AI-CAD Generator.
"""
import os
import cv2
import numpy as np

from cad_engine import (
    generate_spur_gear, generate_stepped_shaft, generate_electric_motor,
    generate_bracket, generate_vehicle_model, evaluate_dfm_feasibility,
    export_model_stl, export_model_obj, export_model_step,
    export_model_dxf, export_costed_boq_excel
)
from vision_pipeline import (
    preprocess_image, classify_input_mode,
    classify_object_and_extract_geometry, create_annotated_cv_visualization
)
from pdf_generator import generate_cad_pdf_report, render_cad_isometric_image


def test_full_suite():
    base_dir = os.path.dirname(os.path.abspath(__file__))

possible_assets = [
    os.path.join(base_dir, "assets"),
    os.path.join(os.path.dirname(base_dir), "assets"),
]

assets_dir = next(
    (path for path in possible_assets if os.path.isdir(path)),
    possible_assets[0]
)
    meta = {
        "doc_id": "CAD-TEST-2026",
        "author": "Simbu T",
        "roll_no": "CS2026-CAP01",
        "department": "Department of Computer Science & Engineering",
        "college": "University College of Engineering",
        "guide": "Dr. A. Sharma, Ph.D.",
        "academic_year": "2025 - 2026"
    }

    test_cases = [
        ("Spur Gear", "sample_gear_drawing.png"),
        ("Stepped Shaft", "sample_shaft_drawing.png"),
        ("Electric Motor", "sample_motor_drawing.png"),
        ("L-Bracket", "sample_bracket_drawing.png"),
        ("Vehicle Body", "sample_vehicle_photo.png")
    ]

    print("=================================================================")
    print("STARTING UNIVERSAL AI-CAD GENERATOR VALIDATION SUITE")
    print("=================================================================")

    # Test Vision Pipeline on all sample images
    for label, fname in test_cases:
        fpath = os.path.join(assets_dir, fname)
        assert os.path.exists(fpath), f"Sample image {fname} missing!"
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cv_d = preprocess_image(img_rgb)
        mode, m_conf = classify_input_mode(img_rgb, cv_d["edges"])
        # If testing vehicle photograph case, ensure photo mode evaluation is also exercised
        eval_mode = "Ordinary Real-World Photograph" if "Vehicle" in label else mode
        cat_d = classify_object_and_extract_geometry(cv_d, eval_mode)
        vis = create_annotated_cv_visualization(img_rgb, cv_d)

        print(f"[{label}] -> Mode: {eval_mode} ({m_conf}%) | Category: {cat_d['category']} ({cat_d['confidence']}%) | Extracted Contours: {len(cv_d['contours'])}")

    print("\n-----------------------------------------------------------------")
    print("TESTING 3D PARAMETRIC SOLID GENERATORS & DFM AUDIT")
    print("-----------------------------------------------------------------")

    models = [
        generate_spur_gear(module=3.0, num_teeth=20, face_width=35.0, bore_dia=25.0),
        generate_stepped_shaft(total_length=300.0, main_dia=50.0, bearing_dia=40.0, end_dia=30.0),
        generate_electric_motor(stator_dia=180.0, stator_len=240.0, shaft_dia=28.0, shaft_len=60.0),
        generate_bracket(arm_vertical=160.0, arm_horizontal=140.0, width=80.0, thickness=18.0),
        generate_vehicle_model(vehicle_type="Passenger Sedan", length_mm=4500.0, width_mm=1800.0, height_mm=1450.0)
    ]

    for model in models:
        dfm = evaluate_dfm_feasibility(model)
        mass = model.get_mass("Structural Steel (AISI 1045 / EN8)")
        print(f"[{model.name}] -> Vol: {model.volume_cm3:.1f} cm3 | Area: {model.surface_area_cm2:.1f} cm2 | Mass: {mass} kg | DFM: {dfm['score']}% ({dfm['rating']})")
        assert model.volume_cm3 > 0, "Volume must be strictly positive"

    print("\n-----------------------------------------------------------------")
    print("TESTING MULTI-CAD EXPORTERS (STL, STEP, OBJ, DXF, XLSX, PDF)")
    print("-----------------------------------------------------------------")

    gear_model = models[0]
    gear_dfm = evaluate_dfm_feasibility(gear_model)

    # 1. STL
    stl_bytes = export_model_stl(gear_model, binary=True)
    assert len(stl_bytes) > 1000, "STL export failed"
    print(f" -> STL (Binary) Export: {len(stl_bytes)} bytes [PASS]")

    # 2. OBJ
    obj_str = export_model_obj(gear_model)
    assert "v " in obj_str and "f " in obj_str, "OBJ export failed"
    print(f" -> Wavefront OBJ Export: {len(obj_str)} chars [PASS]")

    # 3. STEP (ISO 10303)
    step_str = export_model_step(gear_model, meta)
    assert "ISO-10303-21;" in step_str and "END-ISO-10303-21;" in step_str, "STEP export failed"
    print(f" -> STEP AP214 B-Rep Export: {len(step_str)} chars [PASS]")

    # 4. DXF
    dxf_str = export_model_dxf(gear_model)
    assert "SECTION" in dxf_str and "EOF" in dxf_str, "DXF export failed"
    print(f" -> AutoCAD DXF Export: {len(dxf_str)} chars [PASS]")

    # 5. XLSX BoQ
    xlsx_bytes = export_costed_boq_excel(gear_model, gear_dfm, "Structural Steel (AISI 1045 / EN8)", meta)
    assert len(xlsx_bytes) > 2000, "Excel export failed"
    print(f" -> Costed Excel BoQ Export: {len(xlsx_bytes)} bytes [PASS]")

    # 6. ReportLab PDF Dossier
    sample_img_path = os.path.join(assets_dir, "sample_gear_drawing.png")
    img_bgr = cv2.imread(sample_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    cv_d = preprocess_image(img_rgb)
    vis = create_annotated_cv_visualization(img_rgb, cv_d)

    _, in_buf = cv2.imencode(".png", img_bgr)
    _, cv_buf = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    r3d_bytes = render_cad_isometric_image(gear_model.mesh, title=gear_model.name)

    pdf_bytes = generate_cad_pdf_report(
        in_buf.tobytes(), cv_buf.tobytes(), r3d_bytes,
        gear_model, gear_dfm, "Structural Steel (AISI 1045 / EN8)", meta, "Engineering Technical Drawing"
    )
    assert len(pdf_bytes) > 5000, "PDF export failed"
    print(f" -> ReportLab 2-Page Dossier Export: {len(pdf_bytes)} bytes [PASS]")

    print("\n=================================================================")
    print("ALL UNIVERSAL AI-CAD GENERATOR TESTS COMPLETED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_full_suite()
