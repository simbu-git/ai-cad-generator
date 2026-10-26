"""
Verification test to confirm all 5 modules and CAD deliverables execute end-to-end.
"""
import os
import cv2
import numpy as np
from app import (
    detect_and_synthesize_room_features,
    generate_2d_sketch,
    generate_2d_floorplan_schematic,
    build_plotly_3d_wireframe,
    render_3d_isometric_image,
    calculate_technical_blueprint_specs,
    generate_pdf_report,
    generate_obj_model,
    generate_dxf_floorplan,
    generate_excel_boq
)

def test_pipeline():
    sample_img_path = r"C:\Users\simbu T\.gemini\antigravity\scratch\ai-room-reconstruction\assets\sample_bedroom.png"
    assert os.path.exists(sample_img_path), "Sample image missing"

    img_bgr = cv2.imread(sample_img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    room_w, room_l, room_h = 4.5, 5.5, 2.8

    meta = {
        "doc_id": "ARR-SIMBU-2026",
        "author": "Simbu T",
        "roll_no": "CS2026-CAP01",
        "department": "Department of Computer Science & Engineering",
        "college": "University College of Engineering",
        "guide": "Dr. A. Sharma, Ph.D.",
        "academic_year": "2025 - 2026"
    }

    print("Step 1: Testing feature synthesis...")
    feat = detect_and_synthesize_room_features(img_rgb, room_w, room_l, room_h)
    objs = feat["objects"]
    assert len(objs) > 0, "No objects synthesized"
    print(f" -> Synthesized {len(objs)} objects.")

    print("Step 2: Testing Module 1 (OpenCV 2D sketch & floorplan)...")
    sketch_rgb, stats = generate_2d_sketch(img_rgb, 40, 120, 5, 1, "Cyan Architectural Blueprint")
    assert sketch_rgb.shape == img_rgb.shape
    floorplan_rgb = generate_2d_floorplan_schematic(room_w, room_l, objs)
    assert floorplan_rgb.shape[0] > 0
    print(f" -> 2D Sketch ({stats['resolution']}) and Floorplan ({floorplan_rgb.shape}) generated successfully.")

    print("Step 3: Testing Module 2 (Plotly 3D wireframe & Matplotlib 3D render)...")
    fig_3d = build_plotly_3d_wireframe(room_w, room_l, room_h, objs, "Isometric 3D", True)
    assert fig_3d is not None
    render_3d_bytes = render_3d_isometric_image(room_w, room_l, room_h, objs)
    assert len(render_3d_bytes) > 1000
    print(f" -> 3D wireframe model and 3D render ({len(render_3d_bytes)} bytes) created.")

    print("Step 4: Testing Module 3 (Technical blueprint specs & BoQ)...")
    metrics, df_boq = calculate_technical_blueprint_specs(room_w, room_l, room_h, objs)
    assert metrics["floor_area_sqm"] == round(room_w * room_l, 2)
    assert len(df_boq) == len(objs)
    print(f" -> Blueprint specs calculated. Floor area: {metrics['floor_area_sqm']} m2, {len(df_boq)} BoQ items.")

    print("Step 5: Testing Module 4 (ReportLab PDF report generation)...")
    _, in_buf = cv2.imencode(".png", img_bgr)
    _, sk_buf = cv2.imencode(".png", cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR))
    _, fp_buf = cv2.imencode(".png", cv2.cvtColor(floorplan_rgb, cv2.COLOR_RGB2BGR))

    pdf_bytes = generate_pdf_report(
        in_buf.tobytes(), sk_buf.tobytes(), fp_buf.tobytes(), render_3d_bytes,
        metrics, df_boq, meta
    )
    assert len(pdf_bytes) > 5000, "PDF too small"
    print(f" -> PDF generated successfully! Total size: {len(pdf_bytes)} bytes.")

    print("Step 6: Testing Module 5 (CAD & 3D Deliverables)...")
    obj_str = generate_obj_model(room_w, room_l, room_h, objs, meta)
    assert "o Room_Envelope" in obj_str and "v " in obj_str
    print(f" -> 3D Wavefront .OBJ generated ({len(obj_str)} chars).")

    dxf_str = generate_dxf_floorplan(room_w, room_l, objs, meta)
    assert "SECTION" in dxf_str and "EOF" in dxf_str
    print(f" -> AutoCAD .DXF generated ({len(dxf_str)} chars).")

    excel_bytes = generate_excel_boq(df_boq, metrics, meta)
    assert len(excel_bytes) > 3000
    print(f" -> Costed Excel .XLSX generated ({len(excel_bytes)} bytes).")

    print("\nALL MODULES & NEW EXPORT FEATURES PASSED 100%!")

if __name__ == "__main__":
    test_pipeline()
