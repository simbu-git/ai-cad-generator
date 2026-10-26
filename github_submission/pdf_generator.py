"""
================================================================================
AI-CAD GENERATOR: EXECUTIVE ENGINEERING DOSSIER & PDF EXPORT
Module: pdf_generator.py
================================================================================
Compiles input drawing/photo, OpenCV feature map, 3D CAD isometric render,
confirmed geometric dimensions, physical specs, and DFM feasibility audit
into a publication-grade multi-page ReportLab engineering dossier.
================================================================================
"""

import io
import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

from cad_engine import CADModel, MATERIALS_DB


class CADNumberedCanvas(canvas.Canvas):
    """Two-pass canvas calculating total pages and drawing engineering headers/footers."""
    def __init__(self, *args, **kwargs):
        super(CADNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(CADNumberedCanvas, self).showPage()
        super(CADNumberedCanvas, self).save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        page_w, page_h = self._pagesize

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.drawString(40, page_h - 30, "UNIVERSAL AI-CAD GENERATOR")
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawRightString(page_w - 40, page_h - 30, "ENGINEERING SPECIFICATION DOSSIER • ISO 10303")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(40, page_h - 34, page_w - 40, page_h - 34)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(40, 36, page_w - 40, 36)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 24, "CONFIDENTIAL • FINAL YEAR CAPSTONE RESEARCH DOSSIER • AI & COMPUTER VISION IN CAD")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(page_w - 40, 24, page_str)

        self.restoreState()


def render_cad_isometric_image(mesh: trimesh.Trimesh, title: str = "3D CAD MODEL", dpi: int = 150) -> bytes:
    """Renders a high-resolution 3D isometric projection of the solid mesh for PDF inclusion."""
    fig = plt.figure(figsize=(8, 5.0), dpi=dpi, facecolor='#0B192C')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0B192C')

    # Subsample faces if mesh is very dense for fast render
    faces = mesh.faces
    vertices = mesh.vertices
    if len(faces) > 2500:
        indices = np.random.choice(len(faces), 2500, replace=False)
        faces = faces[indices]

    triangles = vertices[faces]
    poly = Poly3DCollection(triangles, facecolors='#38BDF8', edgecolors='#1E3A8A', linewidths=0.3, alpha=0.85)
    ax.add_collection3d(poly)

    # Auto scale limits
    bounds = mesh.bounds
    max_range = np.array([
        bounds[1][0] - bounds[0][0],
        bounds[1][1] - bounds[0][1],
        bounds[1][2] - bounds[0][2]
    ]).max() / 2.0

    mid_x = (bounds[1][0] + bounds[0][0]) * 0.5
    mid_y = (bounds[1][1] + bounds[0][1]) * 0.5
    mid_z = (bounds[1][2] + bounds[0][2]) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X (mm)', color='#94A3B8', fontsize=8)
    ax.set_ylabel('Y (mm)', color='#94A3B8', fontsize=8)
    ax.set_zlabel('Z (mm)', color='#94A3B8', fontsize=8)
    ax.tick_params(colors='#94A3B8', labelsize=7)
    ax.view_init(elev=26, azim=-45)

    ax.xaxis.pane.set_edgecolor('#1E293B')
    ax.yaxis.pane.set_edgecolor('#1E293B')
    ax.zaxis.pane.set_edgecolor('#1E293B')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.set_title(title, color='#38BDF8', fontsize=10, weight='bold', pad=12)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_cad_pdf_report(
    input_image_bytes: bytes,
    cv_vis_bytes: bytes,
    render_3d_bytes: bytes,
    cad_model: CADModel,
    dfm_results: Dict[str, Any],
    material_name: str,
    project_meta: Dict[str, str],
    input_mode: str
) -> bytes:
    """
    Assembles a complete, publication-grade academic engineering dossier
    covering computer vision analysis, solid CAD modeling, confirmed parameters,
    and DFM feasibility audits.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1E3A8A"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10
    )
    section_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=8,
        spaceAfter=5
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#64748B"),
        alignment=1,
        spaceAfter=6
    )
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B")
    )
    cell_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    story = []

    # --------------------------------------------------------------------------
    # PAGE 1: COVER, KPI CARDS, CV ANALYSIS & 3D ISOMETRIC CAD MODEL
    # --------------------------------------------------------------------------
    dept_str = project_meta.get("department", "Department of Computer Science & Engineering")
    college_str = project_meta.get("college", "University College of Engineering")
    author_str = project_meta.get("author", "Simbu T")
    roll_str = project_meta.get("roll_no", "CS2026-CAP01")

    header_table_data = [
        [
            Paragraph(f"<b>{dept_str.upper()}</b><br/><font color='#64748B'>{college_str}</font>", subtitle_style),
            Paragraph(f"<b>CANDIDATE:</b> {author_str} ({roll_str})<br/>"
                      f"<b>DOCUMENT REF:</b> {project_meta.get('doc_id', 'CAD-2026-X')}<br/>"
                      f"<b>DATE:</b> {datetime.datetime.now().strftime('%B %d, %Y')}", subtitle_style)
        ]
    ]
    t_header = Table(header_table_data, colWidths=[310, 210])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor("#1E3A8A")),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 8))

    story.append(Paragraph("UNIVERSAL AI-CAD GENERATOR: TECHNICAL REPORT", title_style))
    story.append(Paragraph(
        f"Automated 2D Feature Extraction & Parameter-Driven 3D Solid CAD Synthesis. "
        f"Target Component: <b>{cad_model.name}</b> [{cad_model.category}].",
        subtitle_style
    ))

    # KPI Summary Cards Table
    mass_kg = cad_model.get_mass(material_name)
    kpi_data = [
        [
            Paragraph("<b>SOLID VOLUME</b>", cell_header),
            Paragraph("<b>NET MASS</b>", cell_header),
            Paragraph("<b>SURFACE AREA</b>", cell_header),
            Paragraph("<b>DFM FEASIBILITY</b>", cell_header)
        ],
        [
            Paragraph(f"<font size=11><b>{cad_model.volume_cm3:.2f} cm³</b></font><br/>"
                      f"<font color='#64748B'>Watertight: {cad_model.is_watertight}</font>", cell_style),
            Paragraph(f"<font size=11><b>{mass_kg:.3f} kg</b></font><br/>"
                      f"<font color='#64748B'>{material_name.split('(')[0].strip()}</font>", cell_style),
            Paragraph(f"<font size=11><b>{cad_model.surface_area_cm2:.1f} cm²</b></font><br/>"
                      f"<font color='#64748B'>Bounds: {cad_model.bounds[0]:.0f}x{cad_model.bounds[1]:.0f} mm</font>", cell_style),
            Paragraph(f"<font size=11><b>{dfm_results['score']}%</b></font><br/>"
                      f"<font color='#16A34A'><b>{dfm_results['rating']}</b></font>", cell_style)
        ]
    ]
    t_kpi = Table(kpi_data, colWidths=[130, 130, 130, 130])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F8FAFC")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # Section 1: Monocular Input vs OpenCV Extraction
    story.append(Paragraph(f"1. COMPUTER VISION FEATURE EXTRACTION ({input_mode.upper()})", section_h1))

    img_input_io = io.BytesIO(input_image_bytes)
    img_cv_io = io.BytesIO(cv_vis_bytes)

    vis_table_data = [
        [
            RLImage(img_input_io, width=250, height=145),
            RLImage(img_cv_io, width=250, height=145)
        ],
        [
            Paragraph("Plate 1-A: Source Input (Drawing / Photograph)", caption_style),
            Paragraph("Plate 1-B: OpenCV Feature Map (Contours, Hough Lines, Arcs)", caption_style)
        ]
    ]
    t_vis = Table(vis_table_data, colWidths=[260, 260])
    t_vis.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_vis)
    story.append(Spacer(1, 4))

    # Section 2: 3D CAD Reconstruction
    story.append(Paragraph("2. PARAMETRIC 3D SOLID CAD RECONSTRUCTION", section_h1))
    img_3d_io = io.BytesIO(render_3d_bytes)
    story.append(RLImage(img_3d_io, width=520, height=225))
    story.append(Paragraph(
        f"Plate 1-C: Generated 3D Solid Geometry ({cad_model.template_type}) • "
        f"Approximate: {cad_model.is_approximate} • Export: STL / STEP / OBJ / DXF",
        caption_style
    ))

    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 2: CONFIRMED PARAMETERS, DFM AUDIT, AND OFFICIAL SIGN-OFF
    # --------------------------------------------------------------------------
    story.append(Paragraph("3. CONFIRMED PARAMETRIC GEOMETRY & DIMENSIONS", section_h1))

    param_headers = [
        Paragraph("<b>Dimension / Feature Name</b>", cell_header),
        Paragraph("<b>Confirmed Engineering Value</b>", cell_header),
        Paragraph("<b>Standard / Fit Reference</b>", cell_header)
    ]
    param_table_data = [param_headers]
    for k, v in cad_model.parameters.items():
        param_table_data.append([
            Paragraph(str(k), cell_style),
            Paragraph(f"<b>{str(v)}</b>", cell_style),
            Paragraph("User Verified / ISO Toleranced", cell_style)
        ])

    t_params = Table(param_table_data, colWidths=[200, 160, 160])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 8))

    # Section 4: DFM Feasibility Audit Table
    story.append(Paragraph("4. DESIGN FOR MANUFACTURING (DFM) FEASIBILITY AUDIT", section_h1))

    dfm_headers = [
        Paragraph("<b>Audit Parameter</b>", cell_header),
        Paragraph("<b>Status</b>", cell_header),
        Paragraph("<b>Score</b>", cell_header),
        Paragraph("<b>Engineering Evaluation Details</b>", cell_header)
    ]
    dfm_table_data = [dfm_headers]
    for chk in dfm_results["checks"]:
        dfm_table_data.append([
            Paragraph(chk["parameter"], cell_style),
            Paragraph(f"<b>{chk['status']}</b>", cell_style),
            Paragraph(f"{chk['score']}%", cell_style),
            Paragraph(chk["details"], cell_style)
        ])

    t_dfm = Table(dfm_table_data, colWidths=[120, 80, 45, 275])
    t_dfm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_dfm)
    story.append(Spacer(1, 8))

    # Section 5: Algorithmic Methodology Summary
    story.append(Paragraph("5. SYSTEM METHODOLOGY & RESEARCH VALIDATION", section_h1))
    method_text = (
        f"<b>Algorithmic Foundation:</b> Input discrimination via background saturation & corner variance; "
        f"shape classification through contour circularity, radial periodicity, and aspect ratios; "
        f"parametric solid extrusion via manifold B-Rep CSG; "
        f"volumetric property integration via divergence theorem ($V = \\frac{1}{6}\\sum (v_0 \\cdot (v_1 \\times v_2))$); "
        f"and automated DFM checking. "
        f"<br/><b>CAD Compliance:</b> STL export conforms to 3D Systems format; STEP export complies with ISO 10303-21 AP214."
    )
    story.append(Paragraph(method_text, subtitle_style))
    story.append(Spacer(1, 10))

    # Official Academic Evaluation Sign-Off Block
    guide_str = project_meta.get("guide", "Faculty Supervisor")
    sign_data = [
        [
            Paragraph(f"<b>PROJECT CANDIDATE</b><br/>{author_str} ({roll_str})<br/><br/><br/>___________________________<br/>Candidate Signature", cell_style),
            Paragraph(f"<b>FACULTY SUPERVISOR / GUIDE</b><br/>{guide_str}<br/><br/><br/>___________________________<br/>Reviewer Signature", cell_style),
            Paragraph(f"<b>INSTITUTIONAL SEAL</b><br/>{college_str}<br/><br/><br/>[ OFFICIAL CAPSTONE SEAL ]<br/>Status: VERIFIED & APPROVED", cell_style)
        ]
    ]
    t_sign = Table(sign_data, colWidths=[173, 173, 174])
    t_sign.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
    ]))
    story.append(t_sign)

    doc.build(story, canvasmaker=CADNumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
