import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime
import re
import json

# Configuration de la page
st.set_page_config(
    page_title="Générateur de Données Minières",
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
        'include_anomalies': False
    }
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

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
    std_values = []  # Initialisation de std_values
    
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
            # Skewness et Kurtosis
            skewness = uploaded_df[element].skew()
            kurtosis = uploaded_df[element].kurtosis()
            
            dist_analysis[element] = {
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

# Fonction pour créer un rapport HTML 
def generate_html_report(data, elements, config, analysis=None):
    """
    Génère un rapport HTML au lieu de PDF
    """
    html = f"""
    <html>
    <head>
        <title>Rapport de Données Minières</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .stats {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>Rapport de Données Minières Synthétiques</h1>
        <p>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        
        <h2>Configuration</h2>
        <table>
            <tr><th>Paramètre</th><th>Valeur</th></tr>
            <tr><td>Nombre d'échantillons</td><td>{config.get('sample_count', 'N/A')}</td></tr>
            <tr><td>Éléments</td><td>{', '.join(elements)}</td></tr>
            <tr><td>Type de distribution</td><td>{config.get('distribution_type', 'N/A')}</td></tr>
            <tr><td>Anomalies</td><td>{"Oui (" + str(config.get('anomaly_percent', 0)) + "%)" if config.get('include_anomalies', False) else "Non"}</td></tr>
            <tr><td>QAQC</td><td>{"Oui" if config.get('include_qaqc', False) else "Non"}</td></tr>
        </table>
        
        <h2>Statistiques Descriptives</h2>
        <div class="stats">
            <table>
                <tr><th>Statistique</th>{"".join([f"<th>{elem}</th>" for elem in elements])}</tr>
    """
    
    # Ajouter les statistiques
    stats = data[elements].describe().reset_index()
    for _, row in stats.iterrows():
        html += f"<tr><td>{row['index']}</td>"
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
            html += f"<td>{formatted_value}</td>"
        html += "</tr>"
    
    html += """
            </table>
        </div>
        
        <h2>Résumé de la distribution</h2>
        <p>Les visualisations détaillées sont disponibles dans l'application.</p>
        
        <h2>Corrélation entre éléments</h2>
        <p>Une matrice de corrélation complète est disponible dans l'application.</p>
    """
    
    # Ajouter une conclusion
    html += """
        <h2>Conclusion</h2>
        <p>Ce rapport résume les statistiques principales des données minières générées. 
           Pour une analyse plus approfondie, utilisez les outils interactifs disponibles dans l'application.</p>
    </body>
    </html>
    """
    
    return html

# Titre principal
st.title("⛏️ Générateur de Données Minières")
st.markdown("Cet outil génère des données synthétiques pour l'industrie minière afin de tester des logiciels ou former des modèles ML.")

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
        index=["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"].index(st.session_state.user_preferences.get('color_theme', 'viridis'))
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
        try:
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
        except Exception as e:
            st.warning(f"Impossible de générer le fichier Excel: {str(e)}. Utilisez le format CSV à la place.")
        
        # Générer HTML au lieu de PDF
        st.subheader("Rapport HTML")
        if st.button("Générer un rapport HTML"):
            with st.spinner("Génération du rapport en cours..."):
                html_report = generate_html_report(all_data, elements, config)
                
                # Créer un lien de téléchargement
                b64_html = base64.b64encode(html_report.encode()).decode()
                href_html = f'<a href="data:text/html;base64,{b64_html}" download="mining_data_report.html">Télécharger le rapport HTML</a>'
                st.markdown(href_html, unsafe_allow_html=True)

with tab_analyze:
    st.header("Analyse des Données")
    
    # Vérifier si des données sont disponibles
    if not st.session_state.data_generated:
        st.info("Veuillez d'abord générer des données dans l'onglet 'Générer'.")
    else:
        st.success("Fonctionnalité avancée disponible dans la version complète.")
        st.info("Cette version simplifiée inclut uniquement la génération de données de base pour assurer la compatibilité maximale avec Streamlit Cloud.")

with tab_compare:
    st.header("Comparaison avec Données Réelles")
    
    # Vérifier si des données sont disponibles
    if not st.session_state.data_generated:
        st.info("Veuillez d'abord générer des données dans l'onglet 'Générer'.")
    else:
        st.success("Fonctionnalité avancée disponible dans la version complète.")
        st.info("Cette version simplifiée inclut uniquement la génération de données de base pour assurer la compatibilité maximale avec Streamlit Cloud.")

with tab_settings:
    st.header("Paramètres et Préférences")
    
    # Préférences générales
    st.subheader("Préférences générales")
    
    # Interface utilisateur
    st.markdown("**Interface utilisateur**")
    
    ui_col1, ui_col2 = st.columns(2)
    
    with ui_col1:
        color_theme = st.selectbox(
            "Thème de couleur par défaut",
            ["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"],
            index=["viridis", "plasma", "inferno", "magma", "cividis", "mako", "rocket", "turbo"].index(st.session_state.user_preferences.get('color_theme', 'viridis'))
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
    
    # Exporter/importer préférences
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
    
    # Informations sur l'application
    st.subheader("À propos de cette application")
    
    st.markdown("""
    ### Générateur de Données Minières
    
    **Version:** 1.0 (version simplifiée)
    
    **Fonctionnalités principales:**
    - Génération de données synthétiques pour l'industrie minière
    - Visualisation de base des données générées
    - Exportation des données en CSV et Excel
    - Rapport HTML basique
    
    **Développé par GeoDataTools**
    
    © 2023 - Tous droits réservés
    """)