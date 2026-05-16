import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image as RLImage, Spacer, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def detect_anomalies(params_actuels, df_history, part_number):
    """
    Module A: Détection d'anomalies de paramètres par rapport à l'historique OK.
    """
    if df_history.empty:
        return
    df_ok = df_history[(df_history['part_number'] == part_number) & (df_history['statut'] == 'OK')]
    if df_ok.empty or len(df_ok) < 3:
        return
    
    anomalies = []
    numeric_params = ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff']
    
    for param in numeric_params:
        if param in params_actuels and params_actuels[param] > 0:
            val_actuelle = params_actuels[param]
            mean_val = df_ok[param].mean()
            std_val = df_ok[param].std()
            
            if pd.isna(std_val) or std_val == 0:
                continue
                
            if val_actuelle > mean_val + 2 * std_val or val_actuelle < mean_val - 2 * std_val:
                pct_diff = ((val_actuelle - mean_val) / mean_val) * 100
                sign = "+" if pct_diff > 0 else ""
                anomalies.append(f"{param} ({sign}{pct_diff:.0f}% vs historique OK — Moy: {mean_val:.1f}, Actuel: {val_actuelle:.1f})")
                
    if anomalies:
        for anomaly in anomalies:
            st.warning(f"⚠️ Paramètre inhabituel : {anomaly}")
    else:
        st.caption("✅ Paramètres dans la plage normale")

def root_cause_analysis(params_actuels, df_history, martensite_pct):
    """
    Module C: Analyse des causes probables pour les rejets.
    """
    if df_history.empty:
        return
    df_ok = df_history[df_history['statut'] == 'OK']
    if df_ok.empty:
        return
    
    numeric_params = ['spindle_speed', 'friction_force', 'upset_force', 'friction_burnoff', 'total_burnoff']
    deviations = []
    
    for param in numeric_params:
        if param in params_actuels and params_actuels[param] > 0:
            val_actuelle = params_actuels[param]
            mean_val = df_ok[param].mean()
            if pd.isna(mean_val) or mean_val == 0:
                continue
                
            pct_diff = ((val_actuelle - mean_val) / mean_val) * 100
            abs_diff = abs(pct_diff)
            if abs_diff > 5: # only report if deviation > 5%
                deviations.append((param, val_actuelle, mean_val, pct_diff, abs_diff))
                
    deviations.sort(key=lambda x: x[4], reverse=True)
    
    if deviations:
        with st.expander("🔍 Analyse des causes probables"):
            st.markdown(f"La martensite est de **{martensite_pct:.2f}%** (Rejet). Voici les paramètres s'éloignant des conditions optimales :")
            for i, (param, val_act, mean_val, pct_diff, _) in enumerate(deviations, 1):
                sign = "+" if pct_diff > 0 else ""
                suggest_range = f"{mean_val*0.95:.1f} - {mean_val*1.05:.1f}"
                st.markdown(f"""
**{i}. {param} : {sign}{pct_diff:.0f}% vs normale**
• Actuel : {val_act:.1f} | Optimal : {mean_val:.1f}
→ **Suggestion** : Ramener vers la plage {suggest_range}
                """)

def compare_best_run(df_history, part_number, IMAGES_DIR):
    """
    Module E: Comparaison visuelle avec le meilleur run de l'historique.
    """
    if df_history.empty:
        return
        
    df_ok = df_history[(df_history['part_number'] == part_number) & (df_history['statut'] == 'OK')]
    if df_ok.empty:
        return
        
    best_run = df_ok.loc[df_ok['martensite_pct'].idxmin()]
    
    image_filename = best_run.get('image_filename')
    if pd.isna(image_filename) or not image_filename:
        return
        
    image_path = os.path.join(IMAGES_DIR, str(image_filename))
    if not os.path.isfile(image_path):
        return
        
    try:
        best_img = Image.open(image_path)
        best_mart = best_run['martensite_pct']
        return best_img, best_mart
    except:
        return None, None

def generate_pdf_report(data_dict, original_pil, annotated_pil, percentages):
    """
    Module B: Génération de rapport PDF.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Header Placeholder (Blue rect)
    header_data = [
        [Paragraph('<font color="white"><b>NEXTEER LOGO</b></font>', styles['Normal']), 
         Paragraph(f'<b>RAPPORT MICROSTRUCTURE</b><br/>{data_dict["timestamp"]}', styles['Normal'])]
    ]
    t_header = Table(header_data, colWidths=[150, 350])
    t_header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#004b87')),
        ('ALIGN', (0,0), (0,0), 'CENTER'),
        ('VALIGN', (0,0), (0,0), 'MIDDLE'),
        ('ALIGN', (1,0), (1,0), 'RIGHT'),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 20))
    
    # Decision
    is_ok = data_dict['statut'] == 'OK'
    decision_text = '<font color="#00C851"><b>CONFORME</b></font>' if is_ok else '<font color="#E31837"><b>NON CONFORME</b></font>'
    story.append(Paragraph(f'<font size=18>{decision_text}</font>', styles['Title']))
    story.append(Spacer(1, 20))
    
    # Identification
    id_data = [
        ['Part Number', 'Opérateur', 'Statut'],
        [data_dict['part_number'], data_dict['operateur'], data_dict['statut']]
    ]
    t_id = Table(id_data, colWidths=[180, 180, 140])
    t_id.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F5F5F5')),
        ('TEXTCOLOR', (2,1), (2,1), colors.green if is_ok else colors.red),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_id)
    story.append(Spacer(1, 20))
    
    # Machine Params
    param_data = [
        ['Spindle Speed', 'Friction Force', 'Upset Force', 'Friction Burnoff', 'Total Burnoff'],
        [f"{data_dict['spindle_speed']} RPM", f"{data_dict['friction_force']} kN", 
         f"{data_dict['upset_force']} kN", f"{data_dict['friction_burnoff']} mm", 
         f"{data_dict['total_burnoff']} mm"]
    ]
    t_param = Table(param_data, colWidths=[100, 100, 100, 100, 100])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F5F5F5')),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_param)
    story.append(Spacer(1, 20))
    
    # Images (Need to save PIL to temporary buffers to use in ReportLab)
    img_width = 240
    img_height = int(img_width * (original_pil.height / original_pil.width))
    
    buf_orig = io.BytesIO()
    original_pil.save(buf_orig, format='PNG')
    buf_orig.seek(0)
    rl_img_orig = RLImage(buf_orig, width=img_width, height=img_height)
    
    buf_ann = io.BytesIO()
    annotated_pil.save(buf_ann, format='PNG')
    buf_ann.seek(0)
    rl_img_ann = RLImage(buf_ann, width=img_width, height=img_height)
    
    img_data = [[rl_img_orig, rl_img_ann], ["Originale", "Segmentation Multi-Phase"]]
    t_img = Table(img_data)
    t_img.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,1), (-1,1), 'Helvetica-Bold'),
    ]))
    story.append(t_img)
    story.append(Spacer(1, 20))
    
    # Phases
    phase_data = [['Phase', '% Détecté']]
    for k, v in percentages.items():
        phase_data.append([k, f"{v:.2f}%"])
        
    t_phase = Table(phase_data, colWidths=[150, 150])
    t_phase.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F5F5F5')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_phase)
    
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
