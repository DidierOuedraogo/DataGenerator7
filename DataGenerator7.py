import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO, StringIO
import base64
from datetime import datetime
import re
import uuid
import json
import tempfile
import os
import time

# Pour les rapports PDF
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-interactif

# Configuration de la page
st.set_page_config(
    page_title="Générateur de Données Minières Avancé",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialiser la session state pour stocker les données et préférences
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'all_data' not in st.session_state:
    st.session_state.all_data = None
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None
if 'crm_df' not in st.session_state:
    st.session_state.crm_df = None
if 'duplicates_df' not in st.session_state:
    st.session_state.duplicates_df = None
if 'blanks_df' not in st.session_state:
    st.session_state.blanks_df = None
if 'elements' not in st.session_state:
    st.session_state.elements = []
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'default_elements': "Au,Cu,Ag,Pb,Zn",
        'default_sample_count': 1000,
        'default_distribution': "Log-normale",
        'color_theme': "viridis",
        'include_qaqc': True,
        'include_anomalies': False,
        'dark_mode': False
    }
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'unique_id' not in st.session_state:
    st.session_state.unique_id = str(uuid.uuid4())[:8]
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# Fonction pour déterminer les valeurs par défaut en fonction de l'élément
def get_default_min(element):
    defaults = {
        "Au": 0.01, "Cu": 10.0, "Ag": 0.5, "Pb": 5.0, "Zn": 5.0, 
        "Fe": 1000.0, "S": 100.0, "As": 1.0, "Sb": 0.5, "Bi": 0.1
    }
    return defaults.get(element, 0.1)

def get_default_max(element):
    defaults = {
        "Au": 30.0, "Cu": 10000.0, "Ag": 100.0, "Pb": 5000.0, "Zn": 10000.0, 
        "Fe": 500000.0, "S": 50000.0, "As": 1000.0, "Sb": 500.0, "Bi": 100.0
    }
    return defaults.get(element, 100.0)

# Fonction pour générer des données d'échantillons
@st.cache_data(ttl=3600, show_spinner=False)
def generate_sample_data(sample_count, elements, min_values, max_values, distribution_type, 
                         correlation_matrix=None, anomaly_percent=0, seed=None):
    """
    Génère des données d'échantillons miniers synthétiques.
    """
    # Fixer la graine aléatoire si fournie
    if seed is not None:
        np.random.seed(seed)
    
    # Initialiser le DataFrame
    data = pd.DataFrame()
    
    # Générer ID et coordonnées
    data['Sample_ID'] = [f'S{i+1:05d}' for i in range(sample_count)]
    data['X'] = np.random.uniform(0, 1000, sample_count)
    data['Y'] = np.random.uniform(0, 1000, sample_count)
    data['Z'] = np.random.uniform(-500, 0, sample_count)
    
    # Générer des dates d'échantillonnage
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    start_timestamp = start_date.timestamp()
    end_timestamp = end_date.timestamp()
    random_timestamps = np.random.uniform(start_timestamp, end_timestamp, sample_count)
    data['Sample_Date'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in random_timestamps]
    
    # Générer des valeurs pour chaque élément
    if distribution_type == "Normal" or distribution_type == "Corrélée":
        # Pour distribution normale ou corrélée
        means = [(min_val + max_val) / 2 for min_val, max_val in zip(min_values, max_values)]
        stds = [(max_val - min_val) / 4 for min_val, max_val in zip(min_values, max_values)]
        
        if distribution_type == "Corrélée" and correlation_matrix is not None:
            # Générer données corrélées à partir de la matrice de corrélation
            cov_matrix = np.zeros((len(elements), len(elements)))
            for i in range(len(elements)):
                for j in range(len(elements)):
                    cov_matrix[i, j] = correlation_matrix[i, j] * stds[i] * stds[j]
            
            element_values = np.random.multivariate_normal(means, cov_matrix, sample_count)
            
            # Assurer que les valeurs restent dans les limites
            for i in range(len(elements)):
                element_values[:, i] = np.clip(element_values[:, i], min_values[i], max_values[i])
            
            for i, element in enumerate(elements):
                data[element] = element_values[:, i]
        else:
            # Distribution normale standard
            for i, element in enumerate(elements):
                data[element] = np.random.normal(means[i], stds[i], sample_count)
                data[element] = np.clip(data[element], min_values[i], max_values[i])
    
    elif distribution_type == "Log-normale":
        # Pour distribution log-normale
        for i, element in enumerate(elements):
            # Paramètres pour la distribution log-normale
            mu = np.log((min_values[i] + max_values[i]) / 2)
            sigma = np.log(max_values[i] / min_values[i]) / 4
            
            data[element] = np.random.lognormal(mu, sigma, sample_count)
            data[element] = np.clip(data[element], min_values[i], max_values[i])
    
    elif distribution_type == "Uniforme":
        # Pour distribution uniforme
        for i, element in enumerate(elements):
            data[element] = np.random.uniform(min_values[i], max_values[i], sample_count)
    
    # Ajouter des anomalies si spécifié
    if anomaly_percent > 0:
        anomaly_count = int(sample_count * anomaly_percent / 100)
        if anomaly_count > 0:
            anomaly_indices = np.random.choice(sample_count, anomaly_count, replace=False)
            
            for i, element in enumerate(elements):
                # Créer des anomalies en multipliant par un facteur aléatoire entre 2 et 5
                anomaly_factor = np.random.uniform(2, 5, anomaly_count)
                data.loc[anomaly_indices, element] *= anomaly_factor
                
                # S'assurer que les anomalies restent dans les limites raisonnables (jusqu'à 2x max_value)
                max_limit = max_values[i] * 2
                data.loc[anomaly_indices, element] = np.clip(data.loc[anomaly_indices, element], 
                                                          min_values[i], 
                                                          max_limit)
    
    # Réinitialiser la graine aléatoire si elle a été fixée
    if seed is not None:
        np.random.seed(None)
        
    return data

# Fonction pour générer des données QAQC
@st.cache_data(ttl=3600, show_spinner=False)
def generate_qaqc_data(sample_data, elements, crm_count=5, duplicate_count=5, blank_count=5, seed=None):
    """
    Génère des données de contrôle qualité (QAQC) basées sur les échantillons.
    """
    # Fixer la graine aléatoire si fournie
    if seed is not None:
        np.random.seed(seed)
        
    # Créer un DataFrame pour stocker toutes les données QAQC
    qaqc_data = pd.DataFrame()
    
    # Générer des échantillons CRM (Certified Reference Material)
    crm_data = pd.DataFrame()
    
    # Déterminer les valeurs de référence pour chaque CRM (pour chaque élément)
    crm_values = []
    std_values = []  # Initialisation de std_values qui manquait dans votre code original
    
    for element in elements:
        # Valeur moyenne de l'élément comme base
        mean_value = sample_data[element].mean()
        # Créer des valeurs CRM distribuées autour de cette moyenne
        element_crm_values = np.linspace(mean_value * 0.5, mean_value * 1.5, crm_count)
        crm_values.append(element_crm_values)
        
        # Écarts-types pour chaque valeur CRM (généralement environ 5-10% de la valeur)
        element_std_values = element_crm_values * np.random.uniform(0.05, 0.1, crm_count)
        std_values.append(element_std_values)
    
    # Créer entries pour chaque CRM et répéter selon le besoin
    crm_repeats = 3  # Nombre de répétitions de chaque CRM
    crm_df = pd.DataFrame()
    
    for i in range(crm_count):
        for j in range(crm_repeats):
            crm_entry = pd.DataFrame({'Sample_ID': [f'CRM-{i+1}-{j+1}']})
            crm_entry['Type'] = 'CRM'
            crm_entry['CRM_ID'] = f'CRM-{i+1}'
            
            # Ajouter des valeurs pour chaque élément
            for k, element in enumerate(elements):
                # Générer une valeur aléatoire depuis distribution normale
                # basée sur la valeur certifiée et l'écart-type
                if crm_values[k][i] > 0:  # Protection contre division par zéro
                    log_std = np.sqrt(np.log(1 + (std_values[k][i]/crm_values[k][i])**2))
                    log_mean = np.log(crm_values[k][i]) - 0.5 * log_std**2
                    crm_entry[element] = np.random.lognormal(log_mean, log_std, 1)[0]
                else:
                    crm_entry[element] = np.max([0, np.random.normal(crm_values[k][i], std_values[k][i], 1)[0]])
            
            crm_df = pd.concat([crm_df, crm_entry], ignore_index=True)
    
    # Générer des duplicates (copies d'échantillons existants avec légère variation)
    sample_indices = np.random.choice(len(sample_data), duplicate_count, replace=False)
    duplicates_df = pd.DataFrame()
    
    for i, idx in enumerate(sample_indices):
        original = sample_data.iloc[idx:idx+1].copy()
        duplicate = original.copy()
        duplicate['Sample_ID'] = f'DUP-{original["Sample_ID"].values[0]}'
        duplicate['Type'] = 'Duplicate'
        duplicate['Original_ID'] = original['Sample_ID'].values[0]
        
        # Ajouter légère variation aux valeurs (±5%)
        for element in elements:
            original_val = original[element].values[0]
            variation = np.random.uniform(-0.05, 0.05) * original_val
            duplicate[element] = max(0, original_val + variation)
        
        duplicates_df = pd.concat([duplicates_df, duplicate], ignore_index=True)
    
    # Générer des blancs (valeurs proches de zéro)
    blanks_df = pd.DataFrame()
    
    for i in range(blank_count):
        blank = pd.DataFrame({'Sample_ID': [f'BLANK-{i+1}']})
        blank['Type'] = 'Blank'
        
        # Pour les blancs, valeurs très faibles (contamination trace)
        for element in elements:
            # Contamination entre 0 et 1% de la limite de détection (estimée comme 1/10 du min)
            detection_limit = sample_data[element].min() / 10
            blank[element] = np.random.uniform(0, 0.01 * detection_limit)
        
        blanks_df = pd.concat([blanks_df, blank], ignore_index=True)
    
    # Combiner tous les types de QAQC
    qaqc_data = pd.concat([crm_df, duplicates_df, blanks_df], ignore_index=True)
    
    # Réinitialiser la graine aléatoire si elle a été fixée
    if seed is not None:
        np.random.seed(None)
    
    return qaqc_data, crm_df, duplicates_df, blanks_df

# Fonction pour analyser les données téléchargées
def analyze_uploaded_data(uploaded_df, elements):
    """
    Analyse un dataframe de données réelles téléchargées pour extraire statistiques et métriques.
    """
    analysis = {}
    
    # Statistiques de base
    analysis['basic_stats'] = uploaded_df[elements].describe()
    
    # Détection des valeurs aberrantes (méthode IQR)
    outliers = {}
    for element in elements:
        if element in uploaded_df.columns:
            Q1 = uploaded_df[element].quantile(0.25)
            Q3 = uploaded_df[element].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = ((uploaded_df[element] < lower_bound) | (uploaded_df[element] > upper_bound)).sum()
            outliers_percent = (outliers_count / len(uploaded_df)) * 100
            outliers[element] = {
                'count': outliers_count,
                'percent': outliers_percent,
                'min': uploaded_df[element].min(),
                'max': uploaded_df[element].max()
            }
    analysis['outliers'] = outliers
    
    # Analyse de la distribution
    dist_analysis = {}
    for element in elements:
        if element in uploaded_df.columns:
            # Test de normalité de Shapiro-Wilk (échantillon de max 5000 points pour performance)
            from scipy import stats
            sample = uploaded_df[element].sample(min(5000, len(uploaded_df)))
            shapiro_test = stats.shapiro(sample)
            
            # Skewness et Kurtosis
            skewness = uploaded_df[element].skew()
            kurtosis = uploaded_df[element].kurtosis()
            
            dist_analysis[element] = {
                'shapiro_p_value': shapiro_test.pvalue,
                'is_normal': shapiro_test.pvalue > 0.05,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'suggested_distribution': 'Log-normale' if skewness > 1.0 else 'Normale'
            }
    analysis['distributions'] = dist_analysis
    
    # Matrice de corrélation
    analysis['correlation'] = uploaded_df[elements].corr()
    
    # Vérification des données QAQC si présentes
    if 'Type' in uploaded_df.columns:
        qaqc_types = uploaded_df['Type'].unique()
        qaqc_counts = uploaded_df['Type'].value_counts().to_dict()
        analysis['qaqc'] = {
            'types_present': list(qaqc_types),
            'counts': qaqc_counts
        }
    
    return analysis

# Fonction pour générer un rapport PDF
def generate_pdf_report(data, elements, config, analysis=None, comparison=None):
    """
    Génère un rapport PDF détaillé des données générées.
    """
    # Créer un fichier temporaire pour le PDF
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_filename = temp_file.name
    temp_file.close()
    
    # Ajouter le fichier à la liste des fichiers temporaires
    st.session_state.temp_files.append(temp_filename)
    
    # Créer un document PDF
    doc = SimpleDocTemplate(temp_filename, pagesize=A4)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    subheading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    # Ajouter titre et date
    story.append(Paragraph("Rapport de Données Minières Synthétiques", title_style))
    story.append(Paragraph(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Configuration utilisée
    story.append(Paragraph("Configuration", heading_style))
    config_data = [
        ["Paramètre", "Valeur"],
        ["Nombre d'échantillons", str(config.get('sample_count', 'N/A'))],
        ["Éléments", ", ".join(elements)],
        ["Type de distribution", config.get('distribution_type', 'N/A')],
        ["Anomalies", f"{config.get('anomaly_percent', 0)}%" if config.get('include_anomalies', False) else "Non"]
    ]
    if config.get('include_qaqc', False):
        config_data.append(["QAQC", "Oui"])
        config_data.append(["Nombre de CRM", str(config.get('crm_count', 'N/A'))])
        config_data.append(["Nombre de duplicatas", str(config.get('duplicate_count', 'N/A'))])
        config_data.append(["Nombre de blancs", str(config.get('blank_count', 'N/A'))])
    else:
        config_data.append(["QAQC", "Non"])
    
    config_table = Table(config_data, colWidths=[2.5*inch, 3*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(config_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Statistiques descriptives
    story.append(Paragraph("Statistiques Descriptives", heading_style))
    
    # Créer une table pour les statistiques
    stats_df = data[elements].describe().reset_index()
    stats_data = [["Statistique"] + elements]
    for _, row in stats_df.iterrows():
        stat_row = [row['index']]
        for element in elements:
            value = row[element]
            # Formater les valeurs numériques
            if isinstance(value, (int, float)):
                if value >= 1000 or value <= 0.001:
                    formatted_value = f"{value:.2e}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            stat_row.append(formatted_value)
        stats_data.append(stat_row)
    
    # Calculer les largeurs de colonnes en fonction du nombre d'éléments
    col_width = 6.5 / (len(elements) + 1)
    col_widths = [1.5*inch] + [col_width*inch] * len(elements)
    
    stats_table = Table(stats_data, colWidths=col_widths)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Visualisations - Histogrammes
    story.append(Paragraph("Distributions des Éléments", heading_style))
    
    for i, element in enumerate(elements):
        story.append(Paragraph(f"Distribution de {element}", subheading_style))
        
        # Créer histogramme
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data[data['Type'] == 'Regular'][element] if 'Type' in data.columns else data[element], 
                    kde=True, ax=ax)
        ax.set_title(f"Distribution de {element}")
        
        # Sauvegarder en fichier temporaire
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_img_name = temp_img.name
        temp_img.close()
        st.session_state.temp_files.append(temp_img_name)
        
        plt.savefig(temp_img_name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Ajouter au PDF
        story.append(Image(temp_img_name, width=6*inch, height=3*inch))
        story.append(Spacer(1, 0.2*inch))
    
    # Matrice de corrélation
    story.append(Paragraph("Matrice de Corrélation", heading_style))
    
    # Calculer la matrice de corrélation
    if 'Type' in data.columns:
        corr_matrix = data[data['Type'] == 'Regular'][elements].corr()
    else:
        corr_matrix = data[elements].corr()
    
    # Créer visualisation de la matrice
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
    
    # Sauvegarder en fichier temporaire
    temp_corr = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_corr_name = temp_corr.name
    temp_corr.close()
    st.session_state.temp_files.append(temp_corr_name)
    
    plt.savefig(temp_corr_name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Ajouter au PDF
    story.append(Image(temp_corr_name, width=6*inch, height=4.5*inch))
    story.append(Spacer(1, 0.3*inch))
    
    # Ajouter section QAQC si incluse
    if config.get('include_qaqc', False) and 'Type' in data.columns:
        story.append(Paragraph("Analyse QAQC", heading_style))
        
        # Résumé QAQC
        qaqc_summary = data['Type'].value_counts().reset_index()
        qaqc_summary.columns = ['Type', 'Count']
        
        qaqc_data = [["Type d'échantillon", "Nombre"]]
        for _, row in qaqc_summary.iterrows():
            qaqc_data.append([row['Type'], str(row['Count'])])
        
        qaqc_table = Table(qaqc_data, colWidths=[3*inch, 2*inch])
        qaqc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ]))
        story.append(qaqc_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Si des données de comparaison sont fournies
        if comparison and isinstance(comparison, dict):
            story.append(Paragraph("Comparaison avec Données Réelles", heading_style))
            
            # Tableau de comparaison des statistiques
            if 'basic_stats' in comparison:
                story.append(Paragraph("Comparaison des Statistiques", subheading_style))
                
                comp_data = [["Statistique", "Type"] + elements]
                stats_generated = data[elements].describe().reset_index()
                stats_real = comparison['basic_stats'].reset_index()
                
                # Combiner les statistiques générées et réelles
                for stat in stats_generated['index'].unique():
                    gen_row = stats_generated[stats_generated['index'] == stat].iloc[0]
                    real_row = stats_real[stats_real['index'] == stat].iloc[0]
                    
                    # Ligne pour données générées
                    gen_values = [stat, "Générées"]
                    for element in elements:
                        value = gen_row[element]
                        if isinstance(value, (int, float)):
                            if value >= 1000 or value <= 0.001:
                                formatted_value = f"{value:.2e}"
                            else:
                                formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                        gen_values.append(formatted_value)
                    comp_data.append(gen_values)
                    
                    # Ligne pour données réelles
                    real_values = [stat, "Réelles"]
                    for element in elements:
                        value = real_row[element]
                        if isinstance(value, (int, float)):
                            if value >= 1000 or value <= 0.001:
                                formatted_value = f"{value:.2e}"
                            else:
                                formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                        real_values.append(formatted_value)
                    comp_data.append(real_values)
                
                # Calculer les largeurs de colonnes
                comp_col_width = 6.5 / (len(elements) + 2)
                comp_col_widths = [1.2*inch, 1*inch] + [comp_col_width*inch] * len(elements)
                
                comp_table = Table(comp_data, colWidths=comp_col_widths)
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (0, -1), colors.lightgrey),
                    ('BACKGROUND', (1, 1), (1, -1), colors.lightblue),
                    ('BOX', (0, 0), (-1, -1), 1, colors.black),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                    ('ALIGN', (2, 1), (-1, -1), 'RIGHT'),
                ]))
                story.append(comp_table)
                story.append(Spacer(1, 0.3*inch))
    
    # Construire le PDF final
    doc.build(story)
    
    return temp_filename

# Fonction pour nettoyer les fichiers temporaires
def cleanup_temp_files():
    for file in st.session_state.temp_files:
        try:
            if os.path.exists(file):
                os.unlink(file)
        except Exception as e:
            st.error(f"Erreur lors du nettoyage des fichiers temporaires: {e}")
    st.session_state.temp_files = []

# Fonction pour sauvegarder les préférences utilisateur
def save_preferences():
    prefs = st.session_state.user_preferences
    prefs_json = json.dumps(prefs)
    b64 = base64.b64encode(prefs_json.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="mining_generator_preferences.json">Télécharger les préférences</a>'
    return href

# Fonction pour charger les préférences utilisateur
def load_preferences(uploaded_file):
    try:
        prefs = json.load(uploaded_file)
        st.session_state.user_preferences = prefs
        return True
    except Exception as e:
        st.error(f"Erreur lors du chargement des préférences: {e}")
        return False

# Partie principale de l'interface
st.title("⛏️ Générateur de Données Minières Avancé")
st.markdown("Cet outil génère des données synthétiques pour l'industrie minière avec fonctionnalités avancées pour l'analyse, la comparaison et le reporting.")

# Onglets pour la navigation principale
tab_generate, tab_analyze, tab_compare, tab_settings = st.tabs(["Générer", "Analyser", "Comparer", "Paramètres"])

with tab_generate:
    # Interface de génération de données
    st.sidebar.header("Paramètres des Données")
    
    # Graine aléatoire pour la reproductibilité
    use_seed = st.sidebar.checkbox("Fixer une graine aléatoire (reproductibilité)", False)
    seed = None
    if use_seed:
        seed = st.sidebar.number_input("Graine aléatoire", 0, 9999, 42)
    
    # Nombre d'échantillons
    sample_count = st.sidebar.slider(
        "Nombre d'échantillons", 
        100, 10000, 
        st.session_state.user_preferences['default_sample_count']
    )
    
    # Paramètres des éléments
    st.sidebar.subheader("Éléments chimiques")
    elements_input = st.sidebar.text_input(
        "Éléments (séparés par des virgules)", 
        st.session_state.user_preferences['default_elements']
    )
    elements = [e.strip() for e in elements_input.split(',')]
    
    # Valeurs min/max pour chaque élément
    min_values = []
    max_values = []
    
    for element in elements:
        st.sidebar.markdown(f"**{element}**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            min_val = st.number_input(f"Min {element}", 0.0, 1000.0, get_default_min(element), step=0.1)
        with col2:
            max_val = st.number_input(f"Max {element}", min_val + 0.1, 10000.0, get_default_max(element), step=0.1)
        min_values.append(min_val)
        max_values.append(max_val)
    
    # Type de distribution
    distribution_type = st.sidebar.selectbox(
        "Type de distribution",
        ["Normal", "Log-normale", "Uniforme", "Corrélée"],
        index=["Normal", "Log-normale", "Uniforme", "Corrélée"].index(st.session_state.user_preferences['default_distribution'])
    )
    
    # Si distribution corrélée, afficher matrice de corrélation
    correlation_matrix = None
    if distribution_type == "Corrélée":
        st.sidebar.subheader("Matrice de corrélation")
        
        # Option pour utiliser une matrice pré-configurée
        use_preset = st.sidebar.checkbox("Utiliser une matrice pré-configurée", False)
        
        if use_preset:
            preset_option = st.sidebar.selectbox(
                "Type de corrélation",
                ["Forte positive", "Modérée positive", "Faible positive", "Anticorrélée", "Mixte"]
            )
            
            # Créer la matrice en fonction du préréglage
            correlation_matrix = np.eye(len(elements))
            
            if preset_option == "Forte positive":
                corr_value = 0.9
            elif preset_option == "Modérée positive":
                corr_value = 0.5
            elif preset_option == "Faible positive":
                corr_value = 0.2
            elif preset_option == "Anticorrélée":
                corr_value = -0.7
            else:  # Mixte
                # Pour mixte, on alterne entre corrélations positives et négatives
                for i in range(len(elements)):
                    for j in range(i+1, len(elements)):
                        if (i + j) % 2 == 0:
                            correlation_matrix[i, j] = 0.7
                            correlation_matrix[j, i] = 0.7
                        else:
                            correlation_matrix[i, j] = -0.4
                            correlation_matrix[j, i] = -0.4
                            
            # Si ce n'est pas mixte, appliquer la même valeur partout
            if preset_option != "Mixte":
                for i in range(len(elements)):
                    for j in range(i+1, len(elements)):
                        correlation_matrix[i, j] = corr_value
                        correlation_matrix[j, i] = corr_value
        else:
            # Interface manuelle pour définir la matrice de corrélation
            correlation_matrix = np.eye(len(elements))
            
            for i in range(len(elements)):
                for j in range(i+1, len(elements)):
                    corr_value = st.sidebar.slider(
                        f"Corrélation {elements[i]}-{elements[j]}", 
                        -1.0, 1.0, 0.0, 0.1
                    )
                    correlation_matrix[i, j] = corr_value
                    correlation_matrix[j, i] = corr_value
    
    # Paramètres QAQC
    st.sidebar.subheader("Paramètres QAQC")
    include_qaqc = st.sidebar.checkbox("Inclure données QAQC", st.session_state.user_preferences['include_qaqc'])
    
    crm_count = 3
    duplicate_count = 3
    blank_count = 3
    
    if include_qaqc:
        qaqc_col1, qaqc_col2 = st.sidebar.columns(2)
        with qaqc_col1:
            crm_count = st.number_input("Nombre de CRM", 1, 10, 3)
            blank_count = st.number_input("Nombre de blancs", 1, 10, 3)
        with qaqc_col2:
            duplicate_count = st.number_input("Nombre de duplicatas", 1, 10, 3)
    
    # Paramètres d'anomalies
    st.sidebar.subheader("Anomalies")
    include_anomalies = st.sidebar.checkbox("Inclure des anomalies", st.session_state.user_preferences['include_anomalies'])
    anomaly_percent = 0
    
    if include_anomalies:
        anomaly_percent = st.sidebar.slider("Pourcentage d'anomalies", 0.1, 10.0, 2.0, 0.1)
    
    # Options de visualisation
    st.sidebar.subheader("Options de visualisation")
    color_theme = st.sidebar.selectbox(
        "Thème de couleur",
        ["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"],
        index=["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"].index(st.session_state.user_preferences['color_theme'])
    )
    
    # Bouton pour mettre à jour les préférences
    if st.sidebar.button("Sauvegarder comme préférences par défaut"):
        st.session_state.user_preferences.update({
            'default_elements': elements_input,
            'default_sample_count': sample_count,
            'default_distribution': distribution_type,
            'color_theme': color_theme,
            'include_qaqc': include_qaqc,
            'include_anomalies': include_anomalies
        })
        st.sidebar.success("Préférences mises à jour!")
    
    # Bouton pour générer les données
    if st.button("Générer les données"):
        # Configuration utilisée pour le rapport
        config = {
            'sample_count': sample_count,
            'distribution_type': distribution_type,
            'include_anomalies': include_anomalies,
            'anomaly_percent': anomaly_percent,
            'include_qaqc': include_qaqc,
            'crm_count': crm_count,
            'duplicate_count': duplicate_count,
            'blank_count': blank_count,
            'color_theme': color_theme
        }
        
        with st.spinner("Génération des données en cours..."):
            # Générer données d'échantillons
            sample_data = generate_sample_data(
                sample_count, elements, min_values, max_values, 
                distribution_type, correlation_matrix, anomaly_percent, seed
            )
            
            # Générer données QAQC si demandé
            if include_qaqc:
                qaqc_data, crm_df, duplicates_df, blanks_df = generate_qaqc_data(
                    sample_data, elements, crm_count, duplicate_count, blank_count, seed
                )
                
                # Ajouter le type aux échantillons normaux
                sample_data['Type'] = 'Regular'
                
                # Combiner les données
                all_data = pd.concat([sample_data, qaqc_data], ignore_index=True)
            else:
                all_data = sample_data
                all_data['Type'] = 'Regular'  # Ajouter quand même le type pour la cohérence
                crm_df = pd.DataFrame()
                duplicates_df = pd.DataFrame()
                blanks_df = pd.DataFrame()
            
            # Stocker les données dans la session state
            st.session_state.all_data = all_data
            st.session_state.sample_data = sample_data
            st.session_state.crm_df = crm_df
            st.session_state.duplicates_df = duplicates_df
            st.session_state.blanks_df = blanks_df
            st.session_state.elements = elements
            st.session_state.data_generated = True
            st.session_state.config = config
        
        # Afficher les données générées
        st.subheader("Aperçu des données générées")
        st.dataframe(all_data.head(10))
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(all_data[elements].describe())
        
        # Visualisations
        st.subheader("Visualisations")
        
        # Utiliser deux colonnes pour les visualisations
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # Histogrammes pour chaque élément
            for element in elements:
                st.markdown(f"**Histogramme de {element}**")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(all_data[all_data['Type'] == 'Regular'][element], kde=True, ax=ax)
                st.pyplot(fig)
        
        with viz_col2:
            # Graphiques de dispersion pour voir la distribution spatiale
            st.markdown("**Distribution spatiale des échantillons**")
            fig = px.scatter_3d(
                all_data[all_data['Type'] == 'Regular'], 
                x='X', y='Y', z='Z',
                color=elements[0],
                color_continuous_scale=color_theme,
                opacity=0.7,
                title=f"Distribution spatiale colorée par {elements[0]}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de corrélation
            st.markdown("**Matrice de corrélation entre éléments**")
            corr_matrix = all_data[all_data['Type'] == 'Regular'][elements].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # QA/QC plots if included
        if include_qaqc:
            st.subheader("Analyse QAQC")
            
            qaqc_col1, qaqc_col2 = st.columns(2)
            
            with qaqc_col1:
                # CRM control charts
                st.markdown("**Graphiques de contrôle CRM**")
                crm_ids = crm_df['CRM_ID'].unique()
                
                for crm_id in crm_ids:
                    crm_subset = crm_df[crm_df['CRM_ID'] == crm_id]
                    for element in elements:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        mean_val = crm_subset[element].mean()
                        std_val = crm_subset[element].std()
                        
                        # Plot CRM values
                        ax.plot(range(len(crm_subset)), crm_subset[element], 'o-', label='Measured')
                        
                        # Add control lines
                        ax.axhline(mean_val, color='green', linestyle='-', label='Mean')
                        ax.axhline(mean_val + 2*std_val, color='orange', linestyle='--', label='+2σ')
                        ax.axhline(mean_val - 2*std_val, color='orange', linestyle='--', label='-2σ')
                        ax.axhline(mean_val + 3*std_val, color='red', linestyle='--', label='+3σ')
                        ax.axhline(mean_val - 3*std_val, color='red', linestyle='--', label='-3σ')
                        
                        ax.set_title(f"{crm_id} - {element}")
                        ax.set_xlabel("Sample Index")
                        ax.set_ylabel(f"{element} Value")
                        ax.legend()
                        st.pyplot(fig)
            
            with qaqc_col2:
                # Duplicate analysis
                st.markdown("**Analyse des duplicatas**")
                
                # Get original samples corresponding to duplicates
                original_ids = duplicates_df['Original_ID'].tolist()
                original_samples = sample_data[sample_data['Sample_ID'].isin(original_ids)]
                
                for element in elements:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    original_values = []
                    duplicate_values = []
                    
                    for _, dup in duplicates_df.iterrows():
                        orig_id = dup['Original_ID']
                        orig_value = original_samples[original_samples['Sample_ID'] == orig_id][element].values[0]
                        original_values.append(orig_value)
                        duplicate_values.append(dup[element])
                    
                    # Scatter plot of original vs duplicate
                    ax.scatter(original_values, duplicate_values)
                    
                    # Add perfect correlation line
                    max_val = max(max(original_values), max(duplicate_values))
                    min_val = min(min(original_values), min(duplicate_values))
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                    
                    # Add ±10% lines
                    ax.plot([min_val, max_val], [min_val*0.9, max_val*0.9], 'g--', label='-10%')
                    ax.plot([min_val, max_val], [min_val*1.1, max_val*1.1], 'g--', label='+10%')
                    
                    ax.set_title(f"{element} Original vs Duplicate")
                    ax.set_xlabel("Original Value")
                    ax.set_ylabel("Duplicate Value")
                    ax.legend()
                    st.pyplot(fig)
                
                # Blank analysis
                st.markdown("**Analyse des blancs**")
                
                # Scatter plot of blanks
                for element in elements:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.scatter(range(len(blanks_df)), blanks_df[element])
                    
                    # Add detection limit line (estimated)
                    detection_limit = sample_data[element].min() / 10
                    ax.axhline(detection_limit, color='red', linestyle='--', label='Detection Limit')
                    
                    ax.set_title(f"{element} Blank Values")
                    ax.set_xlabel("Blank Sample Index")
                    ax.set_ylabel(f"{element} Value")
                    ax.legend()
                    st.pyplot(fig)
        
        # Téléchargement des données
        st.subheader("Télécharger les données")
        
        # Fonction pour créer un lien de téléchargement
        def get_download_link(df, filename, text):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Télécharger {text}</a>'
            return href
        
        # CSV
        st.markdown(get_download_link(all_data, "mining_data.csv", "CSV"), unsafe_allow_html=True)
        
        # Excel
        excel_file = BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            all_data.to_excel(writer, sheet_name='All Data', index=False)
            
            if include_qaqc:
                sample_data.to_excel(writer, sheet_name='Regular Samples', index=False)
                crm_df.to_excel(writer, sheet_name='CRM', index=False)
                duplicates_df.to_excel(writer, sheet_name='Duplicates', index=False)
                blanks_df.to_excel(writer, sheet_name='Blanks', index=False)
        
        excel_file.seek(0)
        b64 = base64.b64encode(excel_file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="mining_data.xlsx">Télécharger XLSX</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Générer PDF
        st.subheader("Rapport PDF")
        if st.button("Générer un rapport PDF"):
            with st.spinner("Génération du rapport en cours..."):
                pdf_file = generate_pdf_report(all_data, elements, config)
                
                # Lire le fichier PDF généré
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()
                
                # Créer un lien de téléchargement
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="mining_data_report.pdf">Télécharger le rapport PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)

with tab_analyze:
    st.header("Analyse Avancée des Données")
    
    # Vérifier si des données sont disponibles
    if not st.session_state.data_generated:
        st.info("Veuillez d'abord générer des données dans l'onglet 'Générer'.")
    else:
        all_data = st.session_state.all_data
        elements = st.session_state.elements
        
        # Filtres et sélection
        st.subheader("Filtrer les données")
        
        # Sélection du type d'échantillons à analyser
        if 'Type' in all_data.columns:
            types = all_data['Type'].unique()
            selected_types = st.multiselect("Types d'échantillons", types, default=list(types))
            filtered_data = all_data[all_data['Type'].isin(selected_types)]
        else:
            filtered_data = all_data
        
        # Filtres sur les coordonnées
        st.markdown("**Filtres spatiaux**")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            x_min, x_max = st.slider("Intervalle X", 
                                    float(all_data['X'].min()), 
                                    float(all_data['X'].max()), 
                                    (float(all_data['X'].min()), float(all_data['X'].max())))
        
        with filter_col2:
            y_min, y_max = st.slider("Intervalle Y", 
                                    float(all_data['Y'].min()), 
                                    float(all_data['Y'].max()), 
                                    (float(all_data['Y'].min()), float(all_data['Y'].max())))
        
        with filter_col3:
            z_min, z_max = st.slider("Intervalle Z", 
                                    float(all_data['Z'].min()), 
                                    float(all_data['Z'].max()), 
                                    (float(all_data['Z'].min()), float(all_data['Z'].max())))
        
        # Appliquer les filtres spatiaux
        filtered_data = filtered_data[
            (filtered_data['X'] >= x_min) & (filtered_data['X'] <= x_max) &
            (filtered_data['Y'] >= y_min) & (filtered_data['Y'] <= y_max) &
            (filtered_data['Z'] >= z_min) & (filtered_data['Z'] <= z_max)
        ]
        
        # Filtres sur les teneurs
        st.markdown("**Filtres sur les teneurs**")
        
        element_filters = st.multiselect("Filtrer par élément", elements)
        
        for element in element_filters:
            elem_min, elem_max = st.slider(f"Intervalle {element}", 
                                        float(all_data[element].min()), 
                                        float(all_data[element].max()), 
                                        (float(all_data[element].min()), float(all_data[element].max())))
            
            filtered_data = filtered_data[
                (filtered_data[element] >= elem_min) & (filtered_data[element] <= elem_max)
            ]
        
        # Afficher les résultats filtrés
        st.markdown(f"**{len(filtered_data)} échantillons** correspondent aux critères de filtrage")
        st.dataframe(filtered_data.head(10))
        
        # Analyses avancées
        st.subheader("Analyses statistiques avancées")
        
        analysis_type = st.selectbox(
            "Type d'analyse",
            ["Distribution univariée", "Analyse bivariée", "Analyse spatiale", "Détection d'anomalies"]
        )
        
        if analysis_type == "Distribution univariée":
            # Sélection de l'élément
            element = st.selectbox("Élément à analyser", elements)
            
            # Sélection du type d'échantillons pour les échantillons normaux
            if 'Type' in filtered_data.columns:
                data_for_analysis = filtered_data[filtered_data['Type'] == 'Regular']
            else:
                data_for_analysis = filtered_data
            
            # Statistiques détaillées
            st.markdown("**Statistiques détaillées**")
            
            from scipy import stats
            
            # Calculer les statistiques
            element_data = data_for_analysis[element].dropna()
            
            mean_val = element_data.mean()
            median_val = element_data.median()
            std_val = element_data.std()
            min_val = element_data.min()
            max_val = element_data.max()
            q1 = element_data.quantile(0.25)
            q3 = element_data.quantile(0.75)
            iqr = q3 - q1
            skewness = element_data.skew()
            kurtosis = element_data.kurtosis()
            
            # Test de normalité
            shapiro_test = stats.shapiro(element_data.sample(min(5000, len(element_data))))
            
            # Afficher les statistiques dans un format lisible
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("Moyenne", f"{mean_val:.4f}")
                st.metric("Écart-type", f"{std_val:.4f}")
                st.metric("CV (%)", f"{(std_val/mean_val*100):.2f}%")
            
            with stats_col2:
                st.metric("Médiane", f"{median_val:.4f}")
                st.metric("Q1 (25%)", f"{q1:.4f}")
                st.metric("Q3 (75%)", f"{q3:.4f}")
            
            with stats_col3:
                st.metric("Min", f"{min_val:.4f}")
                st.metric("Max", f"{max_val:.4f}")
                st.metric("Étendue", f"{max_val-min_val:.4f}")
            
            # Distribution et test de normalité
            dist_col1, dist_col2 = st.columns(2)
            
            with dist_col1:
                st.metric("Skewness", f"{skewness:.4f}")
                st.metric("Kurtosis", f"{kurtosis:.4f}")
                
                # Interprétation de la distribution
                if abs(skewness) < 0.5:
                    st.success("Distribution approximativement symétrique")
                elif abs(skewness) < 1.0:
                    st.info("Distribution modérément asymétrique")
                else:
                    st.warning("Distribution fortement asymétrique")
            
            with dist_col2:
                st.metric("Shapiro p-value", f"{shapiro_test.pvalue:.6f}")
                
                # Interprétation du test de normalité
                if shapiro_test.pvalue > 0.05:
                    st.success("Distribution probablement normale (p > 0.05)")
                else:
                    suggestion = "log-normale" if skewness > 1.0 else "non-normale"
                    st.warning(f"Distribution non normale (p < 0.05), possiblement {suggestion}")
            
            # Visualisations de la distribution
            st.markdown("**Visualisations de la distribution**")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Histogramme avec courbe KDE
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(element_data, kde=True, ax=ax)
                ax.set_title(f"Distribution de {element}")
                ax.set_xlabel(element)
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)
                
                # Box plot
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.boxplot(x=element_data, ax=ax)
                ax.set_title(f"Box plot de {element}")
                ax.set_xlabel(element)
                st.pyplot(fig)
            
            with viz_col2:
                # QQ plot pour tester la normalité
                fig, ax = plt.subplots(figsize=(10, 6))
                stats.probplot(element_data, dist="norm", plot=ax)
                ax.set_title(f"Q-Q Plot de {element} (test de normalité)")
                st.pyplot(fig)
                
                # Histogramme log si skewness élevé
                if skewness > 1.0:
                    # Éviter log(0) en ajoutant une petite valeur si nécessaire
                    log_data = np.log(element_data + (0.01 if min_val <= 0 else 0))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(log_data, kde=True, ax=ax)
                    ax.set_title(f"Distribution log-transformée de {element}")
                    ax.set_xlabel(f"log({element})")
                    ax.set_ylabel("Fréquence")
                    st.pyplot(fig)
        
        elif analysis_type == "Analyse bivariée":
            # Sélection des éléments à comparer
            st.markdown("**Relation entre deux éléments**")
            
            x_element = st.selectbox("Élément X", elements, index=0)
            y_element = st.selectbox("Élément Y", elements, index=min(1, len(elements)-1))
            
            # Sélection du type d'échantillons pour les échantillons normaux
            if 'Type' in filtered_data.columns:
                data_for_analysis = filtered_data[filtered_data['Type'] == 'Regular']
            else:
                data_for_analysis = filtered_data
            
            # Statistiques de corrélation
            st.markdown("**Corrélation entre les éléments**")
            
            corr_col1, corr_col2 = st.columns(2)
            
            with corr_col1:
                # Coefficients de corrélation
                pearson_corr = data_for_analysis[x_element].corr(data_for_analysis[y_element], method='pearson')
                spearman_corr = data_for_analysis[x_element].corr(data_for_analysis[y_element], method='spearman')
                
                st.metric("Corrélation de Pearson", f"{pearson_corr:.4f}")
                st.metric("Corrélation de Spearman (rang)", f"{spearman_corr:.4f}")
                
                # Interprétation de la corrélation
                if abs(pearson_corr) < 0.3:
                    st.info("Corrélation faible")
                elif abs(pearson_corr) < 0.7:
                    st.info("Corrélation modérée")
                else:
                    st.info("Corrélation forte")
            
            with corr_col2:
                # Teste si la corrélation est significative
                from scipy import stats
                
                # Test de Pearson
                pearson_test = stats.pearsonr(data_for_analysis[x_element], data_for_analysis[y_element])
                spearman_test = stats.spearmanr(data_for_analysis[x_element], data_for_analysis[y_element])
                
                st.metric("P-value (Pearson)", f"{pearson_test.pvalue:.6f}")
                st.metric("P-value (Spearman)", f"{spearman_test.pvalue:.6f}")
                
                # Interprétation de la signification
                if pearson_test.pvalue < 0.05:
                    st.success("Corrélation statistiquement significative (p < 0.05)")
                else:
                    st.warning("Corrélation non significative (p > 0.05)")
            
            # Visualisations bivariées
            st.markdown("**Visualisations bivariées**")
            
            biv_col1, biv_col2 = st.columns(2)
            
            with biv_col1:
                # Scatter plot avec ligne de régression
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x=x_element, y=y_element, data=data_for_analysis, ax=ax)
                ax.set_title(f"{x_element} vs {y_element}")
                ax.set_xlabel(x_element)
                ax.set_ylabel(y_element)
                st.pyplot(fig)
            
            with biv_col2:
                # Scatter plot hexbin pour densité (utile pour grands jeux de données)
                if len(data_for_analysis) > 100:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    hb = ax.hexbin(data_for_analysis[x_element], data_for_analysis[y_element], 
                                gridsize=20, cmap='viridis')
                    ax.set_title(f"Densité de points {x_element} vs {y_element}")
                    ax.set_xlabel(x_element)
                    ax.set_ylabel(y_element)
                    plt.colorbar(hb, ax=ax, label='Nombre de points')
                    st.pyplot(fig)
                
                # Joinplot pour combiner distribution et corrélation
                fig = sns.jointplot(x=x_element, y=y_element, data=data_for_analysis, kind='reg',
                                   height=8, ratio=3, space=0.2)
                fig.fig.suptitle(f"Distribution jointe de {x_element} et {y_element}", y=1.05)
                st.pyplot(fig.fig)
            
            # Ajout d'une régression plus avancée
            st.markdown("**Modèle de régression**")
            
            reg_type = st.selectbox(
                "Type de régression",
                ["Linéaire", "Polynomiale", "Logarithmique"]
            )
            
            if reg_type == "Linéaire":
                # Régression linéaire simple
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                
                X = data_for_analysis[x_element].values.reshape(-1, 1)
                y = data_for_analysis[y_element].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Prédictions et métriques
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                # Afficher les résultats
                st.markdown(f"**Équation: {y_element} = {model.coef_[0]:.4f} × {x_element} + {model.intercept_:.4f}**")
                
                reg_col1, reg_col2 = st.columns(2)
                with reg_col1:
                    st.metric("R² (coefficient de détermination)", f"{r2:.4f}")
                with reg_col2:
                    st.metric("RMSE (erreur quadratique moyenne)", f"{rmse:.4f}")
                
                # Visualisation de la régression
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X, y, alpha=0.5)
                ax.plot(X, y_pred, color='red', linewidth=2)
                ax.set_title(f"Régression linéaire: {x_element} vs {y_element}")
                ax.set_xlabel(x_element)
                ax.set_ylabel(y_element)
                st.pyplot(fig)
            
            elif reg_type == "Polynomiale":
                # Régression polynomiale
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                from sklearn.pipeline import make_pipeline
                
                degree = st.slider("Degré du polynôme", 2, 5, 2)
                
                X = data_for_analysis[x_element].values.reshape(-1, 1)
                y = data_for_analysis[y_element].values
                
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model.fit(X, y)
                
                # Prédictions et métriques
                y_pred = model.predict(X)
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                # Afficher les résultats
                st.markdown(f"**Régression polynomiale de degré {degree}**")
                
                reg_col1, reg_col2 = st.columns(2)
                with reg_col1:
                    st.metric("R² (coefficient de détermination)", f"{r2:.4f}")
                with reg_col2:
                    st.metric("RMSE (erreur quadratique moyenne)", f"{rmse:.4f}")
                
                # Visualisation de la régression
                X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_plot = model.predict(X_plot)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X, y, alpha=0.5)
                ax.plot(X_plot, y_plot, color='red', linewidth=2)
                ax.set_title(f"Régression polynomiale: {x_element} vs {y_element}")
                ax.set_xlabel(x_element)
                ax.set_ylabel(y_element)
                st.pyplot(fig)
            
            elif reg_type == "Logarithmique":
                # Régression logarithmique (log-log)
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                
                # Éviter log(0) ou log(négatif)
                valid_indices = (data_for_analysis[x_element] > 0) & (data_for_analysis[y_element] > 0)
                X = np.log(data_for_analysis[x_element][valid_indices].values).reshape(-1, 1)
                y = np.log(data_for_analysis[y_element][valid_indices].values)
                
                if len(X) > 0:
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    # Prédictions et métriques
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    
                    # Afficher les résultats
                    st.markdown(f"**Équation: log({y_element}) = {model.coef_[0]:.4f} × log({x_element}) + {model.intercept_:.4f}**")
                    st.markdown(f"**Forme: {y_element} = {np.exp(model.intercept_):.4f} × {x_element}^{model.coef_[0]:.4f}**")
                    
                    reg_col1, reg_col2 = st.columns(2)
                    with reg_col1:
                        st.metric("R² (coefficient de détermination)", f"{r2:.4f}")
                    with reg_col2:
                        st.metric("RMSE (erreur log-log)", f"{rmse:.4f}")
                    
                    # Visualisation de la régression
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(np.exp(X), np.exp(y), alpha=0.5)
                    
                    # Courbe de régression
                    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_plot = model.predict(X_plot)
                    ax.plot(np.exp(X_plot), np.exp(y_plot), color='red', linewidth=2)
                    
                    ax.set_title(f"Régression logarithmique: {x_element} vs {y_element}")
                    ax.set_xlabel(x_element)
                    ax.set_ylabel(y_element)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    st.pyplot(fig)
                else:
                    st.warning("Impossible d'effectuer une régression logarithmique: certaines valeurs sont négatives ou nulles.")
        
        elif analysis_type == "Analyse spatiale":
            # Analyse de la distribution spatiale
            st.markdown("**Distribution spatiale des teneurs**")
            
            # Sélection de l'élément pour la visualisation
            element = st.selectbox("Élément à visualiser", elements)
            
            # Créer une grille pour la visualisation
            spatial_col1, spatial_col2 = st.columns(2)
            
            with spatial_col1:
                # Carte de chaleur 2D (X-Y)
                st.markdown("**Carte de chaleur X-Y**")
                
                # Créer une figure interactive avec Plotly
                fig = px.density_heatmap(
                    filtered_data, 
                    x='X', 
                    y='Y', 
                    z=element,
                    nbinsx=40, 
                    nbinsy=40,
                    color_continuous_scale=st.session_state.user_preferences['color_theme'],
                    title=f"Distribution spatiale de {element} (vue en plan)"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with spatial_col2:
                # Visualisation 3D
                st.markdown("**Visualisation 3D**")
                
                # Créer une figure 3D interactive
                fig = px.scatter_3d(
                    filtered_data, 
                    x='X', 
                    y='Y', 
                    z='Z',
                    color=element,
                    color_continuous_scale=st.session_state.user_preferences['color_theme'],
                    opacity=0.7,
                    title=f"Distribution 3D de {element}"
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            
            # Sections et profils
            st.markdown("**Sections et profils**")
            
            slice_type = st.radio(
                "Type de section",
                ["Section X", "Section Y", "Profil Z"]
            )
            
            if slice_type == "Section X":
                # Coupe verticale selon X
                x_value = st.slider(
                    "Position X de la section",
                    float(filtered_data['X'].min()),
                    float(filtered_data['X'].max()),
                    (float(filtered_data['X'].min()) + float(filtered_data['X'].max()))/2
                )
                
                # Créer un filtre pour sélectionner les points proches de la section
                tolerance = (filtered_data['X'].max() - filtered_data['X'].min()) / 20
                section_data = filtered_data[
                    (filtered_data['X'] >= x_value - tolerance) &
                    (filtered_data['X'] <= x_value + tolerance)
                ]
                
                # Créer une visualisation de la section
                fig = px.scatter(
                    section_data,
                    x='Y',
                    y='Z',
                    color=element,
                    color_continuous_scale=st.session_state.user_preferences['color_theme'],
                    title=f"Section à X={x_value:.2f}±{tolerance:.2f}",
                    labels={'Y': 'Y', 'Z': 'Z'},
                    height=500
                )
                fig.update_yaxes(autorange="reversed")  # Z positif vers le bas (convention minière)
                st.plotly_chart(fig, use_container_width=True)
            
            elif slice_type == "Section Y":
                # Coupe verticale selon Y
                y_value = st.slider(
                    "Position Y de la section",
                    float(filtered_data['Y'].min()),
                    float(filtered_data['Y'].max()),
                    (float(filtered_data['Y'].min()) + float(filtered_data['Y'].max()))/2
                )
                
                # Créer un filtre pour sélectionner les points proches de la section
                tolerance = (filtered_data['Y'].max() - filtered_data['Y'].min()) / 20
                section_data = filtered_data[
                    (filtered_data['Y'] >= y_value - tolerance) &
                    (filtered_data['Y'] <= y_value + tolerance)
                ]
                
                # Créer une visualisation de la section
                fig = px.scatter(
                    section_data,
                    x='X',
                    y='Z',
                    color=element,
                    color_continuous_scale=st.session_state.user_preferences['color_theme'],
                    title=f"Section à Y={y_value:.2f}±{tolerance:.2f}",
                    labels={'X': 'X', 'Z': 'Z'},
                    height=500
                )
                fig.update_yaxes(autorange="reversed")  # Z positif vers le bas (convention minière)
                st.plotly_chart(fig, use_container_width=True)
            
            elif slice_type == "Profil Z":
                # Coupe horizontale selon Z
                z_value = st.slider(
                    "Élévation Z du profil",
                    float(filtered_data['Z'].min()),
                    float(filtered_data['Z'].max()),
                    (float(filtered_data['Z'].min()) + float(filtered_data['Z'].max()))/2
                )
                
                # Créer un filtre pour sélectionner les points proches du profil
                tolerance = abs(filtered_data['Z'].max() - filtered_data['Z'].min()) / 20
                profile_data = filtered_data[
                    (filtered_data['Z'] >= z_value - tolerance) &
                    (filtered_data['Z'] <= z_value + tolerance)
                ]
                
                # Créer une visualisation du profil
                fig = px.scatter(
                    profile_data,
                    x='X',
                    y='Y',
                    color=element,
                    color_continuous_scale=st.session_state.user_preferences['color_theme'],
                    title=f"Profil à Z={z_value:.2f}±{tolerance:.2f}",
                    labels={'X': 'X', 'Y': 'Y'},
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de variographie simple
            st.markdown("**Analyse de variabilité spatiale**")
            
            if st.checkbox("Afficher l'analyse de variabilité"):
                # Distance moyenne entre points
                from scipy.spatial.distance import pdist
                
                coords = filtered_data[['X', 'Y', 'Z']].values
                distances = pdist(coords)
                avg_dist = np.mean(distances)
                median_dist = np.median(distances)
                min_dist = np.min(distances)
                max_dist = np.max(distances)
                
                st.markdown("**Distances entre échantillons**")
                
                dist_col1, dist_col2, dist_col3 = st.columns(3)
                with dist_col1:
                    st.metric("Distance moyenne", f"{avg_dist:.2f}")
                with dist_col2:
                    st.metric("Distance médiane", f"{median_dist:.2f}")
                with dist_col3:
                    st.metric("Étendue", f"{min_dist:.2f} - {max_dist:.2f}")
                
                # Histogramme des distances
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(distances, bins=30)
                ax.set_title("Distribution des distances entre échantillons")
                ax.set_xlabel("Distance")
                ax.set_ylabel("Fréquence")
                st.pyplot(fig)
                
                # Analyse de continuité simplifiée
                st.markdown("**Continuité spatiale**")
                
                # Calculer un proxy simplifié de la continuité spatiale
                # (différences de valeurs en fonction de la distance)
                if len(filtered_data) > 1 and len(filtered_data) <= 2000:  # Limiter pour performance
                    pairs = []
                    for i in range(len(filtered_data)):
                        for j in range(i+1, min(i+20, len(filtered_data))):  # Limiter le nombre de paires
                            row_i = filtered_data.iloc[i]
                            row_j = filtered_data.iloc[j]
                            
                            # Calculer la distance 3D
                            dist = np.sqrt(
                                (row_i['X'] - row_j['X'])**2 +
                                (row_i['Y'] - row_j['Y'])**2 +
                                (row_i['Z'] - row_j['Z'])**2
                            )
                            
                            # Différence de teneur
                            diff = abs(row_i[element] - row_j[element])
                            
                            pairs.append((dist, diff))
                    
                    # Convertir en DataFrame pour faciliter la visualisation
                    pairs_df = pd.DataFrame(pairs, columns=['Distance', 'Difference'])
                    
                    # Variogramme expérimental simplifié
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(pairs_df['Distance'], pairs_df['Difference'], alpha=0.3)
                    
                    # Ajouter une ligne de tendance
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        pairs_df['Distance'], pairs_df['Difference'])
                    
                    x = np.linspace(pairs_df['Distance'].min(), pairs_df['Distance'].max(), 100)
                    y = slope * x + intercept
                    ax.plot(x, y, color='red', linestyle='--')
                    
                    ax.set_title(f"Variabilité de {element} en fonction de la distance")
                    ax.set_xlabel("Distance entre échantillons")
                    ax.set_ylabel(f"Différence absolue de {element}")
                    st.pyplot(fig)
                    
                    # Interprétation
                    if slope > 0:
                        st.info(f"La variabilité de {element} tend à augmenter avec la distance (pente = {slope:.4f}), "
                               f"ce qui suggère une continuité spatiale.")
                    else:
                        st.info(f"La variabilité de {element} ne montre pas de tendance claire avec la distance "
                               f"(pente = {slope:.4f}), ce qui suggère une faible continuité spatiale.")
                else:
                    st.info("L'analyse de continuité n'est disponible que pour les jeux de données de taille modérée (2-2000 échantillons).")
        
        elif analysis_type == "Détection d'anomalies":
            # Détection d'anomalies dans les données
            st.markdown("**Détection d'anomalies multivariées**")
            
            # Sélection des éléments pour la détection d'anomalies
            elements_for_anomaly = st.multiselect(
                "Éléments à considérer pour la détection d'anomalies",
                elements,
                default=elements
            )
            
            # Sélectionner uniquement les échantillons réguliers pour l'analyse
            if 'Type' in filtered_data.columns:
                data_for_anomaly = filtered_data[filtered_data['Type'] == 'Regular']
            else:
                data_for_anomaly = filtered_data
            
            # Méthode de détection
            anomaly_method = st.selectbox(
                "Méthode de détection",
                ["Z-Score", "IQR (boîte à moustaches)", "Isolation Forest", "DBSCAN"]
            )
            
            if anomaly_method == "Z-Score":
                # Détection basée sur le Z-score
                zscore_threshold = st.slider("Seuil de Z-score", 2.0, 5.0, 3.0, 0.1)
                
                # Calculer les Z-scores pour chaque élément
                anomaly_scores = pd.DataFrame(index=data_for_anomaly.index)
                
                for element in elements_for_anomaly:
                    mean_val = data_for_anomaly[element].mean()
                    std_val = data_for_anomaly[element].std()
                    
                    if std_val > 0:  # Éviter division par zéro
                        anomaly_scores[f'{element}_zscore'] = abs((data_for_anomaly[element] - mean_val) / std_val)
                    else:
                        anomaly_scores[f'{element}_zscore'] = 0
                
                # Calculer le score maximum parmi tous les éléments
                anomaly_scores['max_zscore'] = anomaly_scores.max(axis=1)
                
                # Identifier les anomalies
                anomalies = data_for_anomaly[anomaly_scores['max_zscore'] > zscore_threshold].copy()
                anomalies['Anomaly_Score'] = anomaly_scores.loc[anomalies.index, 'max_zscore']
                
                # Ajouter le score d'anomalie au DataFrame original
                data_with_scores = data_for_anomaly.copy()
                data_with_scores['Anomaly_Score'] = anomaly_scores['max_zscore']
                data_with_scores['Is_Anomaly'] = anomaly_scores['max_zscore'] > zscore_threshold
            
            elif anomaly_method == "IQR (boîte à moustaches)":
                # Détection basée sur la méthode IQR
                iqr_multiplier = st.slider("Multiplicateur IQR", 1.0, 3.0, 1.5, 0.1)
                
                # Calculer les scores d'anomalie pour chaque élément
                anomaly_scores = pd.DataFrame(index=data_for_anomaly.index)
                
                for element in elements_for_anomaly:
                    q1 = data_for_anomaly[element].quantile(0.25)
                    q3 = data_for_anomaly[element].quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr > 0:  # Éviter division par zéro
                        lower_bound = q1 - iqr_multiplier * iqr
                        upper_bound = q3 + iqr_multiplier * iqr
                        
                        # Calculer un score normalisé basé sur la distance aux bornes
                        # 0 si dans les limites, sinon distance normalisée par IQR
                        values = data_for_anomaly[element].values
                        scores = np.zeros_like(values, dtype=float)
                        
                        # Pour les valeurs supérieures à la borne supérieure
                        mask_upper = values > upper_bound
                        if np.any(mask_upper):
                            scores[mask_upper] = (values[mask_upper] - upper_bound) / iqr
                        
                        # Pour les valeurs inférieures à la borne inférieure
                        mask_lower = values < lower_bound
                        if np.any(mask_lower):
                            scores[mask_lower] = (lower_bound - values[mask_lower]) / iqr
                        
                        anomaly_scores[f'{element}_iqr_score'] = scores
                    else:
                        anomaly_scores[f'{element}_iqr_score'] = 0
                
                # Calculer le score maximum parmi tous les éléments
                anomaly_scores['max_iqr_score'] = anomaly_scores.max(axis=1)
                
                # Identifier les anomalies
                anomalies = data_for_anomaly[anomaly_scores['max_iqr_score'] > 0].copy()
                anomalies['Anomaly_Score'] = anomaly_scores.loc[anomalies.index, 'max_iqr_score']
                
                # Ajouter le score d'anomalie au DataFrame original
                data_with_scores = data_for_anomaly.copy()
                data_with_scores['Anomaly_Score'] = anomaly_scores['max_iqr_score']
                data_with_scores['Is_Anomaly'] = anomaly_scores['max_iqr_score'] > 0
            
            elif anomaly_method == "Isolation Forest":
                # Détection basée sur Isolation Forest
                from sklearn.ensemble import IsolationForest
                
                contamination = st.slider("Contamination estimée", 0.01, 0.2, 0.05, 0.01)
                
                # Préparer les données pour l'algorithme
                X = data_for_anomaly[elements_for_anomaly].values
                
                # Ajuster le modèle
                model = IsolationForest(contamination=contamination, random_state=42)
                model.fit(X)
                
                # Prédire les anomalies
                # -1 pour anomalie, 1 pour normal
                y_pred = model.predict(X)
                
                # Calculer les scores d'anomalie
                scores = -model.decision_function(X)  # Négatif pour que les valeurs élevées soient des anomalies
                
                # Identifier les anomalies
                anomalies = data_for_anomaly[y_pred == -1].copy()
                anomalies['Anomaly_Score'] = scores[y_pred == -1]
                
                # Ajouter le score d'anomalie au DataFrame original
                data_with_scores = data_for_anomaly.copy()
                data_with_scores['Anomaly_Score'] = scores
                data_with_scores['Is_Anomaly'] = y_pred == -1
            
            elif anomaly_method == "DBSCAN":
                # Détection basée sur DBSCAN
                from sklearn.cluster import DBSCAN
                from sklearn.preprocessing import StandardScaler
                
                eps = st.slider("Distance maximale (eps)", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Nombre minimal d'échantillons", 3, 20, 5)
                
                # Préparer les données pour l'algorithme
                X = data_for_anomaly[elements_for_anomaly].values
                
                # Standardiser les données
                X_scaled = StandardScaler().fit_transform(X)
                
                # Ajuster le modèle
                model = DBSCAN(eps=eps, min_samples=min_samples)
                y_pred = model.fit_predict(X_scaled)
                
                # Les points avec label -1 sont considérés comme des anomalies
                anomalies = data_for_anomaly[y_pred == -1].copy()
                
                # Calculer une mesure de distance comme score d'anomalie
                from sklearn.neighbors import NearestNeighbors
                
                # Trouver les k plus proches voisins pour calculer une distance moyenne
                k = min(10, len(X_scaled) - 1)
                nn = NearestNeighbors(n_neighbors=k+1)  # +1 car le point lui-même est inclus
                nn.fit(X_scaled)
                distances, _ = nn.kneighbors(X_scaled)
                
                # Utiliser la distance moyenne aux k plus proches voisins comme score d'anomalie
                scores = np.mean(distances[:, 1:], axis=1)  # Exclure le point lui-même (distance 0)
                
                # Normaliser les scores pour faciliter l'interprétation
                if np.std(scores) > 0:
                    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                
                # Ajouter les scores aux anomalies
                anomalies['Anomaly_Score'] = scores[y_pred == -1]
                
                # Ajouter le score d'anomalie au DataFrame original
                data_with_scores = data_for_anomaly.copy()
                data_with_scores['Anomaly_Score'] = scores
                data_with_scores['Is_Anomaly'] = y_pred == -1
            
            # Afficher les résultats
            st.markdown(f"**{len(anomalies)} anomalies détectées** ({len(anomalies)/len(data_for_anomaly)*100:.1f}% des données)")
            
            # Afficher les anomalies triées par score d'anomalie
            if len(anomalies) > 0:
                st.dataframe(anomalies.sort_values(by='Anomaly_Score', ascending=False))
                
                # Visualisation des anomalies
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Distribution des scores d'anomalie
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data_with_scores['Anomaly_Score'], kde=True, ax=ax)
                    if anomaly_method == "Z-Score":
                        ax.axvline(zscore_threshold, color='red', linestyle='--', 
                                 label=f'Seuil ({zscore_threshold})')
                    ax.set_title(f"Distribution des scores d'anomalie ({anomaly_method})")
                    ax.set_xlabel("Score d'anomalie")
                    ax.set_ylabel("Fréquence")
                    if anomaly_method == "Z-Score":
                        ax.legend()
                    st.pyplot(fig)
                
                with viz_col2:
                    # Visualisation 3D avec anomalies en surbrillance
                    if len(elements_for_anomaly) >= 2:
                        # Sélectionner les deux premiers éléments pour la visualisation
                        elem1 = elements_for_anomaly[0]
                        elem2 = elements_for_anomaly[1]
                        
                        fig = go.Figure()
                        
                        # Points normaux
                        fig.add_trace(go.Scatter3d(
                            x=data_with_scores[~data_with_scores['Is_Anomaly']]['X'],
                            y=data_with_scores[~data_with_scores['Is_Anomaly']]['Y'],
                            z=data_with_scores[~data_with_scores['Is_Anomaly']]['Z'],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color='blue',
                                opacity=0.6
                            ),
                            name='Normal'
                        ))
                        
                        # Anomalies
                        fig.add_trace(go.Scatter3d(
                            x=data_with_scores[data_with_scores['Is_Anomaly']]['X'],
                            y=data_with_scores[data_with_scores['Is_Anomaly']]['Y'],
                            z=data_with_scores[data_with_scores['Is_Anomaly']]['Z'],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color='red',
                                opacity=0.8
                            ),
                            name='Anomalie'
                        ))
                        
                        fig.update_layout(
                            title=f"Anomalies détectées dans l'espace 3D",
                            scene=dict(
                                xaxis_title='X',
                                yaxis_title='Y',
                                zaxis_title='Z'
                            ),
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Graphique de dispersion avec les deux principaux éléments
                if len(elements_for_anomaly) >= 2:
                    st.markdown("**Représentation des anomalies dans l'espace des éléments**")
                    
                    # Sélectionner les deux premiers éléments pour la visualisation
                    elem1 = elements_for_anomaly[0]
                    elem2 = elements_for_anomaly[1]
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Points normaux
                    ax.scatter(
                        data_with_scores[~data_with_scores['Is_Anomaly']][elem1],
                        data_with_scores[~data_with_scores['Is_Anomaly']][elem2],
                        color='blue',
                        alpha=0.6,
                        label='Normal'
                    )
                    
                    # Anomalies
                    scatter = ax.scatter(
                        data_with_scores[data_with_scores['Is_Anomaly']][elem1],
                        data_with_scores[data_with_scores['Is_Anomaly']][elem2],
                        color='red',
                        alpha=0.8,
                        label='Anomalie',
                        s=data_with_scores[data_with_scores['Is_Anomaly']]['Anomaly_Score'] * 30 + 30
                    )
                    
                    ax.set_title(f"{elem1} vs {elem2} avec anomalies")
                    ax.set_xlabel(elem1)
                    ax.set_ylabel(elem2)
                    ax.legend()
                    
                    st.pyplot(fig)
                
                # Exporter les résultats
                st.markdown("**Exporter les résultats de détection d'anomalies**")
                
                export_data = data_with_scores.copy()
                
                excel_file = BytesIO()
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    export_data.to_excel(writer, sheet_name='All Data', index=False)
                    anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
                
                excel_file.seek(0)
                b64 = base64.b64encode(excel_file.read()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="anomaly_detection.xlsx">Télécharger les résultats d\'anomalies (XLSX)</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.info("Aucune anomalie détectée avec les paramètres actuels.")

with tab_compare:
    st.header("Comparaison avec Données Réelles")
    
    # Option pour télécharger des données réelles
    st.subheader("Télécharger des données réelles")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Charger les données téléchargées
        try:
            if uploaded_file.name.endswith('.csv'):
                uploaded_data = pd.read_csv(uploaded_file)
            else:
                uploaded_data = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = uploaded_data
            st.session_state.comparison_mode = True
            
            # Afficher un aperçu des données téléchargées
            st.markdown("**Aperçu des données téléchargées**")
            st.dataframe(uploaded_data.head())
            
            # Cartographie des colonnes
            st.subheader("Cartographie des colonnes")
            st.markdown("Associez les colonnes de vos données téléchargées aux colonnes attendues par l'application.")
            
            # Colonnes requises pour la comparaison
            required_cols = ['Sample_ID', 'X', 'Y', 'Z'] + st.session_state.elements
            
            # Créer un mapping pour chaque colonne requise
            col_mapping = {}
            
            # Utiliser des colonnes pour un affichage plus compact
            mapping_cols = st.columns(4)
            
            for i, col in enumerate(required_cols):
                with mapping_cols[i % 4]:
                    col_options = [''] + list(uploaded_data.columns)
                    default_index = 0
                    
                    # Essayer de trouver une correspondance automatique
                    for j, opt in enumerate(col_options):
                        if opt.lower() == col.lower() or opt.lower().replace('_', '') == col.lower().replace('_', ''):
                            default_index = j
                            break
                    
                    col_mapping[col] = st.selectbox(
                        f"Colonne pour {col}",
                        col_options,
                        index=default_index,
                        key=f"map_{col}"
                    )
            
            # Bouton pour effectuer la comparaison
            if st.button("Effectuer la comparaison"):
                # Vérifier si toutes les colonnes requises ont été mappées
                if '' in col_mapping.values():
                    st.error("Toutes les colonnes requises doivent être mappées.")
                else:
                    # Créer un DataFrame mappé
                    mapped_data = pd.DataFrame()
                    
                    for target_col, source_col in col_mapping.items():
                        mapped_data[target_col] = uploaded_data[source_col]
                    
                    # Ajouter Type si disponible
                    if 'Type' in uploaded_data.columns:
                        mapped_data['Type'] = uploaded_data['Type']
                    
                    # Enregistrer les données mappées
                    st.session_state.mapped_data = mapped_data
                    
                    # Analyser les données téléchargées
                    with st.spinner("Analyse des données en cours..."):
                        analysis = analyze_uploaded_data(mapped_data, st.session_state.elements)
                        st.session_state.analysis = analysis
                    
                    # Afficher les résultats de la comparaison
                    st.subheader("Résultats de la comparaison")
                    
                    # Comparer les statistiques descriptives
                    st.markdown("**Comparaison des statistiques descriptives**")
                    
                    # Données générées (si disponibles)
                    if st.session_state.data_generated:
                        generated_data = st.session_state.all_data
                        
                        # Créer un DataFrame pour la comparaison
                        comparison = pd.DataFrame()
                        
                        # Statistiques pour les données générées
                        gen_stats = generated_data[st.session_state.elements].describe()
                        
                        # Statistiques pour les données téléchargées
                        real_stats = mapped_data[st.session_state.elements].describe()
                        
                        # Afficher les statistiques côte à côte
                        comp_col1, comp_col2 = st.columns(2)
                        
                        with comp_col1:
                            st.markdown("**Données générées**")
                            st.dataframe(gen_stats)
                        
                        with comp_col2:
                            st.markdown("**Données réelles**")
                            st.dataframe(real_stats)
                        
                        # Comparer les distributions
                        st.markdown("**Comparaison des distributions**")
                        
                        for element in st.session_state.elements:
                            st.markdown(f"**Distribution de {element}**")
                            
                            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Distribution des données générées
                            sns.histplot(
                                generated_data[generated_data['Type'] == 'Regular'][element] 
                                    if 'Type' in generated_data.columns else generated_data[element],
                                kde=True, 
                                ax=axs[0]
                            )
                            axs[0].set_title(f"Données générées: {element}")
                            
                            # Distribution des données réelles
                            sns.histplot(
                                mapped_data[mapped_data['Type'] == 'Regular'][element] 
                                    if 'Type' in mapped_data.columns else mapped_data[element],
                                kde=True, 
                                ax=axs[1]
                            )
                            axs[1].set_title(f"Données réelles: {element}")
                            
                            st.pyplot(fig)
                        
                        # QQ-plots pour comparer les distributions
                        st.markdown("**QQ-Plots (comparaison quantile-quantile)**")
                        
                        for element in st.session_state.elements:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            
                            # Extraire les données
                            gen_data = generated_data[generated_data['Type'] == 'Regular'][element] if 'Type' in generated_data.columns else generated_data[element]
                            real_data = mapped_data[mapped_data['Type'] == 'Regular'][element] if 'Type' in mapped_data.columns else mapped_data[element]
                            
                            # Calculer les quantiles
                            gen_quantiles = np.quantile(gen_data, np.linspace(0, 1, 100))
                            real_quantiles = np.quantile(real_data, np.linspace(0, 1, 100))
                            
                            # Tracer le QQ-plot
                            ax.scatter(gen_quantiles, real_quantiles)
                            
                            # Ajouter une ligne de référence
                            min_val = min(gen_quantiles.min(), real_quantiles.min())
                            max_val = max(gen_quantiles.max(), real_quantiles.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
                            
                            ax.set_title(f"QQ-Plot pour {element}")
                            ax.set_xlabel("Quantiles des données générées")
                            ax.set_ylabel("Quantiles des données réelles")
                            
                            st.pyplot(fig)
                        
                        # Comparer les corrélations
                        st.markdown("**Comparaison des corrélations**")
                        
                        # Données générées
                        gen_corr = generated_data[generated_data['Type'] == 'Regular'][st.session_state.elements].corr() if 'Type' in generated_data.columns else generated_data[st.session_state.elements].corr()
                        
                        # Données réelles
                        real_corr = mapped_data[mapped_data['Type'] == 'Regular'][st.session_state.elements].corr() if 'Type' in mapped_data.columns else mapped_data[st.session_state.elements].corr()
                        
                        # Afficher les matrices côte à côte
                        corr_col1, corr_col2 = st.columns(2)
                        
                        with corr_col1:
                            st.markdown("**Corrélations des données générées**")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(gen_corr, annot=True, cmap='coolwarm', ax=ax)
                            st.pyplot(fig)
                        
                        with corr_col2:
                            st.markdown("**Corrélations des données réelles**")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(real_corr, annot=True, cmap='coolwarm', ax=ax)
                            st.pyplot(fig)
                        
                        # Calculer la différence entre les matrices
                        corr_diff = gen_corr - real_corr
                        
                        st.markdown("**Différence entre les matrices de corrélation**")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr_diff, annot=True, cmap='coolwarm', ax=ax)
                        ax.set_title("Différence (générée - réelle)")
                        st.pyplot(fig)
                        
                        # Comparer les statistiques spatiales
                        st.markdown("**Comparaison de la distribution spatiale**")
                        
                        # Sélectionner un élément pour la visualisation
                        element = st.selectbox("Élément à visualiser", st.session_state.elements)
                        
                        spatial_col1, spatial_col2 = st.columns(2)
                        
                        with spatial_col1:
                            st.markdown("**Distribution spatiale des données générées**")
                            fig = px.scatter_3d(
                                generated_data[generated_data['Type'] == 'Regular'] if 'Type' in generated_data.columns else generated_data,
                                x='X',
                                y='Y',
                                z='Z',
                                color=element,
                                opacity=0.7,
                                color_continuous_scale=st.session_state.user_preferences['color_theme']
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with spatial_col2:
                            st.markdown("**Distribution spatiale des données réelles**")
                            fig = px.scatter_3d(
                                mapped_data[mapped_data['Type'] == 'Regular'] if 'Type' in mapped_data.columns else mapped_data,
                                x='X',
                                y='Y',
                                z='Z',
                                color=element,
                                opacity=0.7,
                                color_continuous_scale=st.session_state.user_preferences['color_theme']
                            )
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Générer un rapport de comparaison
                        st.subheader("Rapport de comparaison")
                        
                        if st.button("Générer un rapport de comparaison PDF"):
                            with st.spinner("Génération du rapport en cours..."):
                                # Configuration pour le rapport
                                config = st.session_state.config if 'config' in st.session_state else {
                                    'sample_count': len(generated_data),
                                    'distribution_type': 'N/A',
                                    'include_anomalies': False,
                                    'anomaly_percent': 0,
                                    'include_qaqc': 'Type' in generated_data.columns,
                                    'color_theme': st.session_state.user_preferences['color_theme']
                                }
                                
                                # Générer le rapport PDF avec données de comparaison
                                pdf_file = generate_pdf_report(
                                    generated_data, 
                                    st.session_state.elements, 
                                    config, 
                                    analysis=analysis,
                                    comparison={
                                        'basic_stats': real_stats
                                    }
                                )
                                
                                # Lire le fichier PDF généré
                                with open(pdf_file, "rb") as f:
                                    pdf_bytes = f.read()
                                
                                # Créer un lien de téléchargement
                                b64_pdf = base64.b64encode(pdf_bytes).decode()
                                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="mining_data_comparison_report.pdf">Télécharger le rapport de comparaison PDF</a>'
                                st.markdown(href_pdf, unsafe_allow_html=True)
                    else:
                        st.warning("Veuillez d'abord générer des données dans l'onglet 'Générer' pour effectuer une comparaison complète.")
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {e}")

with tab_settings:
    st.header("Paramètres et Préférences")
    
    # Onglets pour les différentes catégories de paramètres
    settings_tab1, settings_tab2, settings_tab3 = st.tabs(["Préférences générales", "Exportation", "Profils utilisateur"])
    
    with settings_tab1:
        st.subheader("Préférences générales")
        
        # Interface utilisateur
        st.markdown("**Interface utilisateur**")
        
        ui_col1, ui_col2 = st.columns(2)
        
        with ui_col1:
            dark_mode = st.checkbox("Mode sombre", st.session_state.user_preferences.get('dark_mode', False))
            st.session_state.user_preferences['dark_mode'] = dark_mode
        
        with ui_col2:
            color_theme = st.selectbox(
                "Thème de couleur par défaut",
                ["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"],
                index=["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"].index(st.session_state.user_preferences['color_theme'])
            )
            st.session_state.user_preferences['color_theme'] = color_theme
        
        # Paramètres de données par défaut
        st.markdown("**Paramètres de données par défaut**")
        
        data_col1, data_col2 = st.columns(2)
        
        with data_col1:
            default_elements = st.text_input(
                "Éléments par défaut (séparés par des virgules)",
                st.session_state.user_preferences['default_elements']
            )
            st.session_state.user_preferences['default_elements'] = default_elements
            
            default_distribution = st.selectbox(
                "Distribution par défaut",
                ["Normal", "Log-normale", "Uniforme", "Corrélée"],
                index=["Normal", "Log-normale", "Uniforme", "Corrélée"].index(st.session_state.user_preferences['default_distribution'])
            )
            st.session_state.user_preferences['default_distribution'] = default_distribution
        
        with data_col2:
            default_sample_count = st.number_input(
                "Nombre d'échantillons par défaut",
                100, 10000, 
                st.session_state.user_preferences['default_sample_count']
            )
            st.session_state.user_preferences['default_sample_count'] = default_sample_count
            
            include_qaqc = st.checkbox(
                "Inclure données QAQC par défaut", 
                st.session_state.user_preferences['include_qaqc']
            )
            st.session_state.user_preferences['include_qaqc'] = include_qaqc
            
            include_anomalies = st.checkbox(
                "Inclure anomalies par défaut", 
                st.session_state.user_preferences['include_anomalies']
            )
            st.session_state.user_preferences['include_anomalies'] = include_anomalies
        
        # Bouton pour sauvegarder les paramètres
        if st.button("Appliquer les préférences"):
            st.success("Préférences mises à jour avec succès!")
    
    with settings_tab2:
        st.subheader("Options d'exportation")
        
        # Format d'exportation préféré
        st.markdown("**Format de fichier préféré**")
        
        export_format = st.radio(
            "Format d'exportation par défaut",
            ["CSV", "Excel (XLSX)", "Les deux"],
            index=2
        )
        st.session_state.user_preferences['export_format'] = export_format
        
        # Options avancées d'exportation
        st.markdown("**Options avancées d'exportation**")
        
        adv_export_col1, adv_export_col2 = st.columns(2)
        
        with adv_export_col1:
            excel_sheets = st.multiselect(
                "Feuilles Excel à inclure par défaut",
                ["Toutes les données", "Échantillons réguliers", "CRM", "Duplicatas", "Blancs", "Statistiques"],
                ["Toutes les données", "Échantillons réguliers", "CRM", "Duplicatas", "Blancs"]
            )
            st.session_state.user_preferences['excel_sheets'] = excel_sheets
        
        with adv_export_col2:
            csv_delimiter = st.selectbox(
                "Délimiteur CSV",
                [",", ";", "Tab"],
                index=0
            )
            st.session_state.user_preferences['csv_delimiter'] = csv_delimiter
            
            include_header = st.checkbox("Inclure en-têtes dans les CSV", True)
            st.session_state.user_preferences['include_header'] = include_header
    
    with settings_tab3:
        st.subheader("Gestion des profils utilisateur")
        
        # Sauvegarder les préférences actuelles
        st.markdown("**Exporter les préférences actuelles**")
        
        st.markdown(save_preferences(), unsafe_allow_html=True)
        
        # Charger des préférences
        st.markdown("**Importer des préférences**")
        
        pref_file = st.file_uploader("Charger un fichier de préférences", type=["json"])
        
        if pref_file is not None:
            if load_preferences(pref_file):
                st.success("Préférences chargées avec succès!")
        
        # Réinitialiser les préférences
        st.markdown("**Réinitialiser les préférences**")
        
        if st.button("Restaurer les paramètres par défaut"):
            st.session_state.user_preferences = {
                'default_elements': "Au,Cu,Ag,Pb,Zn",
                'default_sample_count': 1000,
                'default_distribution': "Log-normale",
                'color_theme': "viridis",
                'include_qaqc': True,
                'include_anomalies': False,
                'dark_mode': False
            }
            st.success("Préférences réinitialisées aux valeurs par défaut!")
    
    # Informations sur l'application
    st.subheader("À propos de cette application")
    
    st.markdown("""
    ### Générateur de Données Minières Avancé
    
    **Version:** 2.0
    
    **Fonctionnalités principales:**
    - Génération de données synthétiques pour l'industrie minière
    - Analyse statistique avancée et détection d'anomalies
    - Comparaison avec des données réelles téléchargées
    - Génération de rapports PDF détaillés
    - Personnalisation complète des préférences utilisateur
    
    **Développé par GeoDataTools**
    
    **Contact:** contact@geodatatools.com
    
    **Site web:** www.geodatatools.com
    
    © 2025 - Tous droits réservés
    """)

# Nettoyer les fichiers temporaires au moment de quitter l'application
atexit.register(cleanup_temp_files)