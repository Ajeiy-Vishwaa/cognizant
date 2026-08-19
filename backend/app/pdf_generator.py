import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as ReportLabImage,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


def generate_claim_report_pdf(
    output_path,
    customer_id,
    customer_email,
    execution_time,
    original_image_path,
    annotated_image_path,
    gradcam_path,
    fraud_label,
    fraud_probability,
    model_confidence,
    damage_result,
):
    """Generates the vehicle claim assessment PDF report."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ClaimReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=4,
        textColor=colors.HexColor("#1A2B4C"),
    )

    subtitle_style = ParagraphStyle(
        "ClaimReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        spaceAfter=14,
        textColor=colors.grey,
    )

    section_style = ParagraphStyle(
        "ClaimReportSection",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor("#1A2B4C"),
    )

    body_style = ParagraphStyle(
        "ClaimReportBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Vehicle Claim Assessment Report",
    )

    story = []

    # Title & Header
    story.append(Paragraph("VEHICLE CLAIM ASSESSMENT REPORT", title_style))
    story.append(Paragraph("AI-assisted vehicle insurance claim assessment", subtitle_style))

    # Section 1: Customer & Claim Information
    story.append(Paragraph("1. CUSTOMER & CLAIM INFORMATION", section_style))

    claim_rows = [
        ["Customer ID", str(customer_id)],
        ["Customer Email", str(customer_email)],
        ["Assessment Date", str(execution_time)],
    ]

    claim_table = Table(claim_rows, colWidths=[50 * mm, 120 * mm])
    claim_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D3D3D3")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(claim_table)

    # Section 2: Fraud Assessment
    story.append(Paragraph("2. FRAUD ASSESSMENT", section_style))

    is_fraud = str(fraud_label).lower() in ["fraud", "flagged for fraud", "true"]
    status_color = colors.HexColor("#CC0000") if is_fraud else colors.HexColor("#008000")

    status_paragraph = Paragraph(
        f"<b><font color='{status_color.hexval()}'>{str(fraud_label).upper()}</font></b>",
        body_style,
    )

    fraud_rows = [
        ["Classification", status_paragraph],
        ["Fraud Probability", f"{float(fraud_probability):.2f}%"],
        ["Model Confidence", f"{float(model_confidence):.2f}%"],
    ]

    fraud_table = Table(fraud_rows, colWidths=[50 * mm, 120 * mm])
    fraud_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D3D3D3")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(fraud_table)

    # Section 3: Damage Assessment
    story.append(Paragraph("3. DAMAGE ASSESSMENT", section_style))

    damage_result = damage_result or {}
    damage_count = int(damage_result.get("damage_count", 0))
    damage_detected = bool(damage_result.get("damage_detected", False)) or (damage_count > 0)
    damage_coverage = float(damage_result.get("damage_coverage_percentage", 0.0))
    overall_severity = str(damage_result.get("overall_severity", "N/A"))
    overall_score = float(damage_result.get("overall_severity_score", 0.0))

    damage_rows = [
        ["Damage Detected", "YES" if damage_detected else "NO"],
        ["Damage Regions", str(damage_count)],
        ["Damage Coverage", f"{damage_coverage:.2f}%"],
        ["Overall Severity", overall_severity],
        ["Severity Score", f"{overall_score:.2f}/100"],
    ]

    damage_table = Table(damage_rows, colWidths=[50 * mm, 120 * mm])
    damage_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D3D3D3")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F5F5")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(damage_table)

    # Section 4: Visual Evidence
    story.append(Paragraph("4. VISUAL EVIDENCE", section_style))

    evidence = []
    for label, path in [
        ("Original Image", original_image_path),
        ("YOLO Damage Detection", annotated_image_path),
        ("ConvNeXt Grad-CAM", gradcam_path),
    ]:
        if path and Path(path).exists():
            evidence.append(
                [
                    Paragraph(
                        label,
                        ParagraphStyle(
                            "EvidenceLabel",
                            parent=body_style,
                            alignment=TA_CENTER,
                            fontName="Helvetica-Bold",
                            fontSize=7.5,
                        ),
                    ),
                    ReportLabImage(
                        str(path),
                        width=105 * mm,
                        height=60 * mm,
                    ),
                ]
            )

    if evidence:
        evidence_table = Table(evidence, colWidths=[38 * mm, 120 * mm])
        evidence_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D3D3D3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (1, 0), (1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(evidence_table)

    doc.build(story)
    return output_path