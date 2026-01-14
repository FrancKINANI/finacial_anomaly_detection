"""
Module de génération de rapports PDF professionnels
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
import io

def generate_report(predictions_history, dataset=None, include_graphs=True, include_details=True):
    """
    Génère un rapport PDF professionnel
    
    Args:
        predictions_history (list): Historique des prédictions
        dataset (pd.DataFrame): Dataset utilisé (optionnel)
        include_graphs (bool): Inclure les graphiques
        include_details (bool): Inclure les détails techniques
    
    Returns:
        bytes: Contenu du PDF
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Style personnalisé
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Centré
    )
    
    # Page de garde
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("📊 RAPPORT DE PRÉDICTION", title_style))
    story.append(Paragraph("Détection d'Anomalies Financières", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"Nombre de prédictions: {len(predictions_history)}", styles['Normal']))
    story.append(PageBreak())
    
    # Résumé exécutif
    story.append(Paragraph("RÉSUMÉ EXÉCUTIF", styles['Heading1']))
    story.append(Spacer(1, 0.2*inch))
    
    # Statistiques
    total_preds = len(predictions_history)
    bankruptcies = sum(1 for p in predictions_history if p['prediction'] == 1)
    healthy = total_preds - bankruptcies
    
    summary_data = [
        ['Métrique', 'Valeur'],
        ['Total d\'analyses', str(total_preds)],
        ['Entreprises saines', f"{healthy} ({healthy/total_preds*100:.1f}%)"],
        ['Risques détectés', f"{bankruptcies} ({bankruptcies/total_preds*100:.1f}%)"],
        ['Probabilité moyenne', f"{sum(p['probability'] for p in predictions_history)/total_preds*100:.1f}%"]
    ]
    
    t = Table(summary_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*inch))
    
    # Détails des prédictions
    if include_details:
        story.append(Paragraph("DÉTAILS DES PRÉDICTIONS", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        for idx, pred in enumerate(predictions_history, 1):
            # En-tête de prédiction
            story.append(Paragraph(f"Prédiction #{idx}", styles['Heading2']))
            story.append(Paragraph(f"Timestamp: {pred['timestamp'].strftime('%H:%M:%S')}", styles['Normal']))
            
            # Résultat
            result = "⚠️ RISQUE DE FAILLITE" if pred['prediction'] == 1 else "✅ ENTREPRISE SAINE"
            result_color = colors.red if pred['prediction'] == 1 else colors.green
            
            result_style = ParagraphStyle(
                'Result',
                parent=styles['Normal'],
                fontSize=14,
                textColor=result_color,
                fontName='Helvetica-Bold'
            )
            story.append(Paragraph(result, result_style))
            story.append(Paragraph(f"Probabilité: {pred['probability']*100:.2f}%", styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
    
    # Graphiques
    if include_graphs and len(predictions_history) > 0:
        story.append(PageBreak())
        story.append(Paragraph("VISUALISATIONS", styles['Heading1']))
        story.append(Spacer(1, 0.2*inch))
        
        # Créer graphique de distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Camembert
        labels = ['Saines', 'Risque']
        sizes = [healthy, bankruptcies]
        colors_pie = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution des Prédictions')
        
        # Barplot des probabilités
        probabilities = [p['probability'] for p in predictions_history]
        ax2.hist(probabilities, bins=10, color='steelblue', edgecolor='black')
        ax2.set_xlabel('Probabilité de Faillite')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution des Probabilités')
        
        plt.tight_layout()
        
        # Sauvegarder dans buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Ajouter au PDF
        img = RLImage(img_buffer, width=6*inch, height=2.4*inch)
        story.append(img)
    
    # Footer
    story.append(PageBreak())
    story.append(Spacer(1, 6*inch))
    footer_text = """
    <para align=center>
    <b>Détection d'Anomalies Financières</b><br/>
    Application développée avec Python, Streamlit et Machine Learning<br/>
    © 2026 - Institut Supérieur de Management
    </para>
    """
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Générer le PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_simple_report(prediction_result):
    """
    Génère un rapport simple pour une seule prédiction
    
    Args:
        prediction_result (dict): Résultat de prédiction
    
    Returns:
        bytes: Contenu du PDF
    """
    return generate_report([prediction_result], include_graphs=False, include_details=True)