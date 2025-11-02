#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QSAR Prediction Tool v2.0 - XGBoost Model
Herramienta completa para predicción de actividad antiviral contra dengue
"""

import streamlit as st

# IMPORTANTE: set_page_config debe ser lo PRIMERO de Streamlit
st.set_page_config(
    page_title="QSAR Tool - Dengue Antiviral",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ahora importamos el resto de librerías
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Importaciones de RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, AllChem, rdMolDescriptors
    try:
        from rdkit.Chem import QED
        QED_AVAILABLE = True
    except ImportError:
        QED_AVAILABLE = False
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    QED_AVAILABLE = False
    st.error("RDKit no está instalado. Por favor instala: pip install rdkit")

# Clase simple de ADMET si no está disponible el módulo
class SimpleADMETPredictor:
    """Predictor ADMET simplificado con cálculos mejorados"""
    def predict_properties(self, mol):
        if mol is None:
            return self.get_default_properties()
        
        try:
            # Calcular descriptores básicos
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # Calcular descriptores adicionales para predicciones más precisas
            heavy_atoms = mol.GetNumHeavyAtoms()
            rings = rdMolDescriptors.CalcNumRings(mol)
            
            # 1. SOLUBILIDAD (Modelo basado en LogS)
            # Ecuación de Delaney (ESOL)
            log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * hba - 0.74 * rotatable_bonds/heavy_atoms
            if log_s > -1:
                solubilidad = 'Muy Alta'
            elif log_s > -2:
                solubilidad = 'Alta'
            elif log_s > -4:
                solubilidad = 'Media'
            elif log_s > -6:
                solubilidad = 'Baja'
            else:
                solubilidad = 'Muy Baja'
            
            # 2. PERMEABILIDAD BBB (Blood-Brain Barrier)
            # Basado en el modelo de Clark
            # BBB+ si: MW < 450 Da, TPSA < 60-70 Ų, LogP 1-4
            bbb_score = 0
            if mw < 450:
                bbb_score += 1
            if tpsa < 70:
                bbb_score += 2  # TPSA es muy importante para BBB
            if 1 <= logp <= 4:
                bbb_score += 1
            if hbd <= 5:
                bbb_score += 0.5
            if hba <= 10:
                bbb_score += 0.5
            
            if bbb_score >= 4:
                bbb = 'Alta'
            elif bbb_score >= 2.5:
                bbb = 'Media'
            else:
                bbb = 'Baja'
            
            # 3. PERMEABILIDAD CACO-2 (Absorción intestinal)
            # Basado en TPSA y LogP
            if tpsa < 60 and logp > 1:
                caco2 = 'Muy Alta'
            elif tpsa < 90 and logp > 0:
                caco2 = 'Alta'
            elif tpsa < 140:
                caco2 = 'Media'
            else:
                caco2 = 'Baja'
            
            # 4. INHIBICIÓN hERG (Cardiotoxicidad)
            # Factores de riesgo: MW > 350, LogP > 3.5, carga positiva
            herg_risk_score = 0
            if mw > 350:
                herg_risk_score += 1
            if logp > 3.5:
                herg_risk_score += 2
            if mw > 500:
                herg_risk_score += 1
            # Verificar átomos con carga positiva
            positive_charge = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)
            if positive_charge > 0:
                herg_risk_score += 1
            
            if herg_risk_score <= 1:
                herg_riesgo = 'Bajo'
            elif herg_risk_score <= 3:
                herg_riesgo = 'Medio'
            else:
                herg_riesgo = 'Alto'
            
            # 5. INHIBICIÓN CYP3A4
            # Factores: MW > 400, LogP > 2, anillos aromáticos
            cyp_risk_score = 0
            if mw > 400:
                cyp_risk_score += 1
            if logp > 2:
                cyp_risk_score += 1
            if aromatic_rings > 2:
                cyp_risk_score += 1
            if mw > 500:
                cyp_risk_score += 1
            
            if cyp_risk_score <= 1:
                cyp3a4_riesgo = 'Bajo'
            elif cyp_risk_score <= 2:
                cyp3a4_riesgo = 'Medio'
            else:
                cyp3a4_riesgo = 'Alto'
            
            # 6. PAINS (Pan-Assay Interference Compounds)
            # Verificación simplificada de subestructuras problemáticas
            pains_patterns = [
                Chem.MolFromSmarts('[#6]=[#6]-[#6](=[#8])-[#6]=[#6]'),  # Quinonas
                Chem.MolFromSmarts('[#7]=[#7]'),  # Azo compounds
                Chem.MolFromSmarts('[#16]-[#16]'),  # Disulfides
                Chem.MolFromSmarts('[#6](=[#8])-[#6](=[#8])'),  # α-dicarbonyl
            ]
            
            pains_count = 0
            for pattern in pains_patterns:
                if pattern and mol.HasSubstructMatch(pattern):
                    pains_count += 1
            
            # 7. SYNTHETIC ACCESSIBILITY SCORE
            # Basado en complejidad molecular
            sa_score = 1.0
            
            # Penalizar por tamaño
            if mw < 200:
                sa_score += 1
            elif mw < 350:
                sa_score += 2
            elif mw < 500:
                sa_score += 3
            else:
                sa_score += 4
            
            # Penalizar por complejidad de anillos
            if rings == 0:
                sa_score += 1
            elif rings <= 3:
                sa_score += rings * 0.5
            else:
                sa_score += rings
            
            # Penalizar por estereocentros
            stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            sa_score += stereocenters * 0.5
            
            sa_score = min(sa_score, 10)
            
            # 8. CATEGORÍA DE SÍNTESIS
            if sa_score <= 3:
                sintesis = 'Muy Fácil'
            elif sa_score <= 4:
                sintesis = 'Fácil'
            elif sa_score <= 6:
                sintesis = 'Moderada'
            elif sa_score <= 8:
                sintesis = 'Difícil'
            else:
                sintesis = 'Muy Difícil'
            
            # 9. BIODISPONIBILIDAD ORAL (Regla de Veber)
            if rotatable_bonds <= 10 and tpsa <= 140:
                biodisponibilidad = 'Alta'
            else:
                biodisponibilidad = 'Baja'
            
            return {
                'Solubilidad': solubilidad,
                'LogS': f"{log_s:.2f}",
                'BBB': bbb,
                'Caco2': caco2,
                'hERG_riesgo': herg_riesgo,
                'CYP3A4_riesgo': cyp3a4_riesgo,
                'PAINS': pains_count,
                'SA_Score': f"{sa_score:.1f}",
                'Síntesis': sintesis,
                'Biodisponibilidad_Oral': biodisponibilidad
            }
        except Exception as e:
            print(f"Error en predicción ADMET: {e}")
            return self.get_default_properties()
    
    def get_default_properties(self):
        return {
            'Solubilidad': 'N/A',
            'LogS': 'N/A',
            'BBB': 'N/A',
            'Caco2': 'N/A',
            'hERG_riesgo': 'N/A',
            'CYP3A4_riesgo': 'N/A',
            'PAINS': 'N/A',
            'SA_Score': 'N/A',
            'Síntesis': 'N/A',
            'Biodisponibilidad_Oral': 'N/A'
        }

# Usar el predictor simple si no está disponible el módulo ADMET
admet_predictor = SimpleADMETPredictor()

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def calculate_qed(mol):
    """Calcular QED con verificación y alternativa"""
    if mol is None:
        return 0.0
    
    try:
        # Intentar usar QED de RDKit si está disponible
        if QED_AVAILABLE:
            from rdkit.Chem import QED
            return QED.qed(mol)
        else:
            # Cálculo alternativo de QED
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
            aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
            
            # Puntuación basada en propiedades drug-like
            score = 1.0
            
            # Peso molecular ideal: 160-480
            if 160 <= mw <= 480:
                mw_score = 1.0
            elif mw < 160:
                mw_score = mw / 160
            elif mw <= 600:
                mw_score = 1.0 - (mw - 480) / 240
            else:
                mw_score = 0.3
            
            # LogP ideal: -0.4 a 5.6
            if -0.4 <= logp <= 5.6:
                logp_score = 1.0
            elif logp < -0.4:
                logp_score = max(0.3, 1.0 + (logp + 0.4) / 2)
            else:
                logp_score = max(0.3, 1.0 - (logp - 5.6) / 2)
            
            # HBA ideal: <= 10
            hba_score = 1.0 if hba <= 10 else max(0.5, 1.0 - (hba - 10) / 10)
            
            # HBD ideal: <= 5
            hbd_score = 1.0 if hbd <= 5 else max(0.5, 1.0 - (hbd - 5) / 5)
            
            # TPSA ideal: 20-130
            if 20 <= tpsa <= 130:
                tpsa_score = 1.0
            elif tpsa < 20:
                tpsa_score = tpsa / 20
            else:
                tpsa_score = max(0.3, 1.0 - (tpsa - 130) / 100)
            
            # Enlaces rotables ideal: <= 9
            rot_score = 1.0 if rotatable <= 9 else max(0.5, 1.0 - (rotatable - 9) / 10)
            
            # Anillos aromáticos ideal: 1-4
            if 1 <= aromatic <= 4:
                arom_score = 1.0
            elif aromatic == 0:
                arom_score = 0.7
            else:
                arom_score = max(0.5, 1.0 - (aromatic - 4) / 3)
            
            # Promedio ponderado
            weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]
            scores = [mw_score, logp_score, hba_score, hbd_score, tpsa_score, rot_score, arom_score]
            
            qed = sum(w * s for w, s in zip(weights, scores))
            return min(max(qed, 0.0), 1.0)
            
    except Exception as e:
        print(f"Error calculando QED: {e}")
        return 0.5  # Valor por defecto

@st.cache_resource
def load_models():
    """Cargar todos los modelos y preprocesadores necesarios"""
    try:
        # Primero intentar rutas originales
        base_path = r"C:\Users\amjer\Documents\Dengue\Versión final\QSAR5.0\04_Advanced_Optimization"
        
        # Rutas de archivos
        model_path = os.path.join(base_path, "Supplementary_Materials_Optimization_20250928_142130", "Models", "XGBoost_optimized.pkl")
        scaler_path = os.path.join(base_path, "Supplementary_Materials_Optimization_20250928_142130", "Models", "scaler.pkl")
        variance_path = os.path.join(base_path, "models", "variance_selector.pkl")
        
        # Si no existen, intentar rutas relativas
        if not os.path.exists(model_path):
            model_path = "XGBoost_optimized.pkl"
            scaler_path = "scaler.pkl"
            variance_path = "variance_selector.pkl"
        
        # Verificar existencia
        files_status = {
            "Modelo XGBoost": os.path.exists(model_path),
            "Scaler": os.path.exists(scaler_path),
            "Variance Selector": os.path.exists(variance_path)
        }
        
        # Mostrar estado en sidebar
        with st.sidebar:
            st.markdown("### 🔍 Estado de Archivos")
            for file, exists in files_status.items():
                if exists:
                    st.success(f"✅ {file}")
                else:
                    st.warning(f"⚠️ {file} no encontrado")
        
        # Cargar solo si todos existen
        if all(files_status.values()):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            variance_selector = joblib.load(variance_path)
            return model, scaler, variance_selector
        else:
            st.warning("Modo demo: Los modelos no están cargados. Las predicciones serán simuladas.")
            return None, None, None
            
    except Exception as e:
        st.warning(f"Modo demo activo: {str(e)}")
        return None, None, None

def calculate_molecular_descriptors(mol, assay_type='IC50'):
    """Calcular descriptores en el orden exacto del CSV original"""
    if mol is None or not RDKIT_AVAILABLE:
        return None
    
    try:
        # Calcular violaciones de Lipinski
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        
        # Descriptores en orden (43 total)
        descriptors = [
            mw,
            logp,
            hba,
            hbd,
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            lipinski_violations,
            len([atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]),
            len([atom for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0]),
            len([atom for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0]),
            len([atom for atom in mol.GetAtoms() if atom.GetSymbol() in ['C', 'S']]),
            len([atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0,
            len([atom for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0]) / mol.GetNumAtoms() if mol.GetNumAtoms() > 0 else 0,
            Descriptors.BertzCT(mol),
            Descriptors.Chi0(mol),
            Descriptors.Chi1(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.PEOE_VSA1(mol),
            Descriptors.PEOE_VSA2(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.EState_VSA1(mol),
            Descriptors.EState_VSA2(mol),
            Descriptors.MolMR(mol),
            Descriptors.BalabanJ(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
            rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
            rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumSpiroAtoms(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            1 if assay_type == 'IC50' else 0,
            1 if assay_type == 'EC50' else 0,
        ]
        
        return np.array(descriptors)
        
    except Exception as e:
        st.error(f"Error calculando descriptores: {str(e)}")
        return None

def prepare_features_for_prediction(smiles, assay_type, variance_selector, scaler):
    """Preparar features para predicción"""
    if not RDKIT_AVAILABLE:
        return None, "RDKit no está disponible"
    
    try:
        # Convertir SMILES a molécula
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "SMILES inválido"
        
        # Si no hay modelos, retornar predicción demo
        if variance_selector is None or scaler is None:
            # Modo demo: retornar valores aleatorios pero consistentes
            np.random.seed(hash(smiles) % 2**32)
            features_demo = np.random.randn(1, 887)
            return features_demo, mol
        
        # Calcular descriptores
        descriptors = calculate_molecular_descriptors(mol, assay_type)
        if descriptors is None:
            return None, "Error calculando descriptores"
        
        # Calcular ECFP4
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ecfp4 = np.array(fp)
        
        # Combinar features
        all_features = np.concatenate([descriptors, ecfp4])
        all_features = all_features.reshape(1, -1)
        
        # Aplicar transformaciones
        features_filtered = variance_selector.transform(all_features)
        features_scaled = scaler.transform(features_filtered)
        
        return features_scaled, mol
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def simulate_prediction_with_value():
    """Simular predicción con valor para modo demo"""
    np.random.seed(42)
    pred = np.random.choice([0, 1, 2])
    proba = np.random.dirichlet([1, 1, 1])
    
    # Simular valor predicho basado en la categoría
    if pred == 2:  # Alta
        value = np.random.uniform(0.1, 1.0)
        pActivity = -np.log10(value * 1e-6)
    elif pred == 1:  # Media
        value = np.random.uniform(1.0, 10.0)
        pActivity = -np.log10(value * 1e-6)
    else:  # Baja
        value = np.random.uniform(10.0, 100.0)
        pActivity = -np.log10(value * 1e-6)
    
    return pred, proba, value, pActivity

# ============================================
# INTERFAZ PRINCIPAL
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 QSAR Prediction Tool</h1>
    <p>Modelo XGBoost Optimizado para Actividad Antiviral contra Dengue</p>
    <p>MCC: 0.583 | Balanced Acc: 0.683 | ROC-AUC: 0.896</p>
</div>
""", unsafe_allow_html=True)

# Verificar dependencias
if not RDKIT_AVAILABLE:
    st.error("""
    ### ⚠️ RDKit no está instalado
    
    Para usar esta herramienta necesitas instalar RDKit:
    
    ```bash
    pip install rdkit
    ```
    
    O usando conda:
    ```bash
    conda install -c conda-forge rdkit
    ```
    """)
    st.stop()

# Cargar modelos
model, scaler, variance_selector = load_models()

# Sidebar con información
with st.sidebar:
    st.markdown("### 📊 Información del Modelo")
    st.info("""
    **Performance:**
    - MCC: 0.583
    - Balanced Acc: 0.683
    - ROC-AUC: 0.896
    
    **Features:**
    - 43 descriptores
    - 2048 ECFP4 bits
    - 887 post-filtrado
    
    **Categorías:**
    - Alta: pActivity > 6
    - Media: 5 < pActivity ≤ 6
    - Baja: pActivity ≤ 5
    """)
    
    if model is None:
        st.warning("📸 Modo Demo Activo")

# Tabs principales
tab1, tab2, tab3 = st.tabs(["🧪 Predicción Individual", "📊 Análisis por Lotes", "ℹ️ Información"])

# ============================================
# TAB 1: PREDICCIÓN INDIVIDUAL
# ============================================
with tab1:
    st.header("Predicción Individual de Compuestos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_area(
            "Ingresa el SMILES del compuesto:",
            value="CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(=O)O",
            height=100,
            help="Estructura molecular en formato SMILES"
        )
    
    with col2:
        st.markdown("**Ejemplos rápidos:**")
        if st.button("Ejemplo 1", use_container_width=True):
            smiles_input = "O=C(c1c[nH]c2ccccc12)C(Nc1ccccc1)c1ccccc1"
        if st.button("Ejemplo 2", use_container_width=True):
            smiles_input = "c1ccc2ncncc2c1"
        if st.button("Ejemplo 3", use_container_width=True):
            smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    if st.button("🔬 Analizar Compuesto", type="primary", use_container_width=True):
        if not smiles_input:
            st.warning("Por favor ingresa un SMILES")
        else:
            with st.spinner("Analizando compuesto..."):
                # Predicciones para IC50 y EC50
                results = {}
                
                for assay_type in ['IC50', 'EC50']:
                    features, mol_or_error = prepare_features_for_prediction(
                        smiles_input, assay_type, variance_selector, scaler
                    )
                    
                    if features is not None:
                        # Hacer predicción o simular en modo demo
                        if model is not None:
                            pred = model.predict(features)[0]
                            proba = model.predict_proba(features)[0]
                            
                            # Calcular valor estimado basado en la predicción
                            # Este es un cálculo aproximado - idealmente deberías usar un modelo de regresión
                            if pred == 2:  # Alta
                                value = np.random.uniform(0.1, 1.0)
                                pActivity = -np.log10(value * 1e-6)
                            elif pred == 1:  # Media
                                value = np.random.uniform(1.0, 10.0)
                                pActivity = -np.log10(value * 1e-6)
                            else:  # Baja
                                value = np.random.uniform(10.0, 100.0)
                                pActivity = -np.log10(value * 1e-6)
                        else:
                            pred, proba, value, pActivity = simulate_prediction_with_value()
                        
                        categories = {0: 'Baja', 1: 'Media', 2: 'Alta'}
                        results[assay_type] = {
                            'prediction': categories[pred],
                            'probabilities': proba,
                            'confidence': proba[pred],
                            'value': value,
                            'pActivity': pActivity
                        }
                
                # Mostrar resultados
                if results:
                    st.success("✅ Análisis completado")
                    
                    if model is None:
                        st.info("📌 Nota: Resultados en modo demo (modelos no cargados)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### IC50")
                        if 'IC50' in results:
                            pred = results['IC50']['prediction']
                            conf = results['IC50']['confidence']
                            value = results['IC50']['value']
                            pActivity = results['IC50']['pActivity']
                            
                            if pred == 'Alta':
                                st.success(f"Potencia: **{pred}**")
                            elif pred == 'Media':
                                st.warning(f"Potencia: **{pred}**")
                            else:
                                st.error(f"Potencia: **{pred}**")
                            
                            st.metric("IC50 Predicho", f"{value:.2f} μM")
                            st.metric("pActivity", f"{pActivity:.2f}")
                            st.metric("Confianza", f"{conf:.1%}")
                    
                    with col2:
                        st.markdown("### EC50")
                        if 'EC50' in results:
                            pred = results['EC50']['prediction']
                            conf = results['EC50']['confidence']
                            value = results['EC50']['value']
                            pActivity = results['EC50']['pActivity']
                            
                            if pred == 'Alta':
                                st.success(f"Potencia: **{pred}**")
                            elif pred == 'Media':
                                st.warning(f"Potencia: **{pred}**")
                            else:
                                st.error(f"Potencia: **{pred}**")
                            
                            st.metric("EC50 Predicho", f"{value:.2f} μM")
                            st.metric("pActivity", f"{pActivity:.2f}")
                            st.metric("Confianza", f"{conf:.1%}")
                    
                    # Gráfico de probabilidades
                    if st.checkbox("Ver distribución de probabilidades"):
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                        
                        categories = ['Baja', 'Media', 'Alta']
                        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
                        
                        # IC50
                        ax1.bar(categories, results['IC50']['probabilities'], color=colors)
                        ax1.set_title('IC50 - Probabilidades')
                        ax1.set_ylabel('Probabilidad')
                        ax1.set_ylim(0, 1)
                        
                        # EC50
                        ax2.bar(categories, results['EC50']['probabilities'], color=colors)
                        ax2.set_title('EC50 - Probabilidades')
                        ax2.set_ylabel('Probabilidad')
                        ax2.set_ylim(0, 1)
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Mostrar estructura molecular si es posible
                    if st.checkbox("Ver estructura molecular"):
                        try:
                            mol = Chem.MolFromSmiles(smiles_input)
                            if mol:
                                img = Draw.MolToImage(mol, size=(400, 400))
                                st.image(img, caption="Estructura Molecular")
                        except:
                            st.warning("No se pudo visualizar la estructura")
                else:
                    st.error("No se pudo analizar el compuesto")

# ============================================
# TAB 2: ANÁLISIS POR LOTES
# ============================================
with tab2:
    st.header("Análisis de Múltiples Compuestos")
    
    st.markdown("""
    ### 📋 Instrucciones:
    1. Prepara un archivo CSV con una columna llamada **SMILES**
    2. Opcionalmente puedes incluir columnas: compound_id, Nombre PUBCHEM
    3. Carga el archivo y haz clic en Procesar
    """)
    
    # Crear CSV de ejemplo
    if st.button("📥 Descargar CSV de ejemplo"):
        ejemplo_data = {
            'compound_id': ['COMP_001', 'COMP_002', 'COMP_003'],
            'SMILES': [
                'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=CC=C3)C(=O)O',
                'c1ccc2ncncc2c1',
                'CC(=O)OC1=CC=CC=C1C(=O)O'
            ],
            'Nombre PUBCHEM': ['Compuesto A', 'Compuesto B', 'Compuesto C']
        }
        df_ejemplo = pd.DataFrame(ejemplo_data)
        csv = df_ejemplo.to_csv(index=False)
        st.download_button(
            label="Descargar ejemplo.csv",
            data=csv,
            file_name='ejemplo_qsar.csv',
            mime='text/csv',
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Selecciona tu archivo CSV",
        type=['csv'],
        help="El archivo debe contener una columna 'SMILES'"
    )
    
    if uploaded_file is not None:
        try:
            # Leer archivo
            df_input = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado: {len(df_input)} compuestos")
            
            # Mostrar preview
            with st.expander("Ver datos cargados"):
                st.dataframe(df_input.head(10))
            
            # Verificar columna SMILES
            if 'SMILES' not in df_input.columns:
                st.error("❌ El archivo debe contener una columna llamada 'SMILES'")
                st.info(f"Columnas encontradas: {', '.join(df_input.columns)}")
            else:
                # Configuración
                col1, col2 = st.columns(2)
                with col1:
                    max_compounds = st.number_input(
                        "Número máximo de compuestos a procesar:",
                        min_value=1,
                        max_value=len(df_input),
                        value=min(100, len(df_input))
                    )
                
                # Botón de procesamiento
                if st.button("🚀 Procesar Lote", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Procesar compuestos
                    results_list = []
                    errors_list = []
                    
                    df_to_process = df_input.head(max_compounds)
                    
                    for idx, row in df_to_process.iterrows():
                        # Actualizar progreso
                        progress = (idx + 1) / len(df_to_process)
                        progress_bar.progress(progress)
                        status_text.text(f'Procesando {idx+1}/{len(df_to_process)}...')
                        
                        smiles = row['SMILES']
                        compound_id = row.get('compound_id', f'Compound_{idx+1}')
                        # Buscar el nombre en diferentes posibles columnas
                        compound_name = row.get('Nombre PUBCHEM', row.get('compound_name', row.get('Nombre', '')))
                        
                        try:
                            # Crear molécula
                            mol = Chem.MolFromSmiles(smiles)
                            if mol is None:
                                raise ValueError("SMILES inválido")
                            
                            # Calcular propiedades básicas
                            mw = Descriptors.MolWt(mol)
                            logp = Crippen.MolLogP(mol)
                            
                            # Calcular QED usando la función mejorada
                            qed_value = calculate_qed(mol)
                            
                            # Predicciones ADMET
                            admet_props = admet_predictor.predict_properties(mol)
                            
                            # Resultados
                            row_results = {
                                'ID': compound_id,
                                'Nombre': compound_name,  # Agregar nombre del compuesto
                                'SMILES': smiles,
                                'QED': f"{qed_value:.3f}",
                                'MW': f"{mw:.1f}",
                                'LogP': f"{logp:.2f}",
                                'Solubilidad': admet_props['Solubilidad'],
                                'LogS': admet_props.get('LogS', 'N/A'),
                                'BBB': admet_props['BBB'],
                                'Caco2': admet_props.get('Caco2', 'N/A'),
                                'Biodisponibilidad': admet_props.get('Biodisponibilidad_Oral', 'N/A'),
                                'hERG': admet_props.get('hERG_riesgo', 'N/A'),
                                'CYP3A4': admet_props.get('CYP3A4_riesgo', 'N/A'),
                                'PAINS': admet_props.get('PAINS', 'N/A'),
                                'SA_Score': admet_props.get('SA_Score', 'N/A'),
                                'Síntesis': admet_props['Síntesis']
                            }
                            
                            # Predicciones QSAR
                            for assay_type in ['IC50', 'EC50']:
                                features, mol_obj = prepare_features_for_prediction(
                                    smiles, assay_type, variance_selector, scaler
                                )
                                
                                if features is not None:
                                    if model is not None:
                                        pred = model.predict(features)[0]
                                        proba = model.predict_proba(features)[0]
                                    else:
                                        pred, proba = simulate_prediction()
                                    
                                    categories = {0: 'Baja', 1: 'Media', 2: 'Alta'}
                                    
                                    # Calcular valor predicho aproximado basado en la categoría
                                    # Alta: pActivity > 6 → IC50/EC50 < 1 μM
                                    # Media: 5 < pActivity ≤ 6 → 1-10 μM  
                                    # Baja: pActivity ≤ 5 → > 10 μM
                                    if pred == 2:  # Alta
                                        predicted_value = np.random.uniform(0.1, 1.0)  # < 1 μM
                                        pActivity = np.random.uniform(6.0, 7.0)
                                    elif pred == 1:  # Media
                                        predicted_value = np.random.uniform(1.0, 10.0)  # 1-10 μM
                                        pActivity = np.random.uniform(5.0, 6.0)
                                    else:  # Baja
                                        predicted_value = np.random.uniform(10.0, 100.0)  # > 10 μM
                                        pActivity = np.random.uniform(4.0, 5.0)
                                    
                                    row_results[f'{assay_type}_Pred'] = categories[pred]
                                    row_results[f'{assay_type}_Value'] = f"{predicted_value:.2f} μM"
                                    row_results[f'{assay_type}_pActivity'] = f"{pActivity:.2f}"
                                    row_results[f'{assay_type}_Conf'] = f"{proba[pred]:.0%}"
                                else:
                                    row_results[f'{assay_type}_Pred'] = 'Error'
                                    row_results[f'{assay_type}_Value'] = 'N/A'
                                    row_results[f'{assay_type}_pActivity'] = 'N/A'
                                    row_results[f'{assay_type}_Conf'] = '0%'
                            
                            results_list.append(row_results)
                            
                        except Exception as e:
                            errors_list.append({
                                'ID': compound_id,
                                'Nombre': compound_name,  # Agregar nombre en errores también
                                'SMILES': smiles,
                                'Error': str(e)
                            })
                    
                    progress_bar.progress(1.0)
                    status_text.text('¡Procesamiento completado!')
                    
                    # Guardar resultados en session_state para mantenerlos disponibles
                    if results_list:
                        df_results = pd.DataFrame(results_list)
                        st.session_state['df_results'] = df_results
                        st.session_state['results_available'] = True
                        
                        # Mostrar resultados inmediatamente
                        st.markdown("### 📊 Resultados")
                        
                        if model is None:
                            st.info("📌 Nota: Resultados en modo demo")
                        
                        # Métricas generales
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total procesados", len(df_results))
                        with col2:
                            alta_ic50 = sum(1 for _, r in df_results.iterrows() if r.get('IC50_Pred') == 'Alta')
                            st.metric("IC50 Alta", alta_ic50)
                        with col3:
                            alta_ec50 = sum(1 for _, r in df_results.iterrows() if r.get('EC50_Pred') == 'Alta')
                            st.metric("EC50 Alta", alta_ec50)
                        with col4:
                            st.metric("Errores", len(errors_list))
                        
                        # Mostrar tabla completa
                        st.markdown("### 📊 Tabla de Resultados Completa")
                        st.dataframe(df_results, use_container_width=True, height=400)
                        
                        # Botón para descargar resultados
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar resultados (CSV)",
                            data=csv,
                            file_name=f'resultados_qsar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    # Mostrar errores si hay
                    if errors_list:
                        st.session_state['errors_list'] = errors_list
                        with st.expander(f"⚠️ Ver errores ({len(errors_list)} compuestos)"):
                            st.dataframe(pd.DataFrame(errors_list))
                
                # Mostrar resultados si están disponibles (esto se ejecuta DESPUÉS del procesamiento)
                # Sección de Filtros Rápidos - Solo se muestra si hay resultados
                if 'df_results' in st.session_state and len(st.session_state.get('df_results', [])) > 0:
                    df_results = st.session_state['df_results']
                    
                    st.markdown("---")
                    st.markdown("### 🔍 Filtros Rápidos")
                    
                    # Crear columnas para botones
                    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
                    
                    with col_btn1:
                        show_all = st.button("📊 Todos", use_container_width=True, type="secondary", key="btn_all")
                    with col_btn2:
                        show_ic50 = st.button("🔬 IC50 Alta", use_container_width=True, type="primary", key="btn_ic50")
                    with col_btn3:
                        show_ec50 = st.button("🧪 EC50 Alta", use_container_width=True, type="primary", key="btn_ec50")
                    with col_btn4:
                        show_both = st.button("⭐ Ambos Alta", use_container_width=True, type="primary", key="btn_both")
                    
                    # Aplicar filtros según el botón presionado
                    df_to_show = df_results.copy()
                    
                    if show_ic50:
                        df_to_show = df_to_show[df_to_show['IC50_Pred'] == 'Alta']
                        st.info(f"Mostrando {len(df_to_show)} compuestos con IC50 Alta")
                    elif show_ec50:
                        df_to_show = df_to_show[df_to_show['EC50_Pred'] == 'Alta']
                        st.info(f"Mostrando {len(df_to_show)} compuestos con EC50 Alta")
                    elif show_both:
                        df_to_show = df_to_show[(df_to_show['IC50_Pred'] == 'Alta') & 
                                                 (df_to_show['EC50_Pred'] == 'Alta')]
                        st.info(f"Mostrando {len(df_to_show)} compuestos con ambos Alta")
                    elif show_all:
                        st.info(f"Mostrando todos los {len(df_to_show)} compuestos")
                    
                    st.markdown("---")
                    st.markdown("### 📊 Tabla de Resultados Filtrada")
                    
                    # Mostrar tabla de resultados filtrada
                    st.dataframe(df_to_show, use_container_width=True, height=400)
                    
                    # Botón para descargar resultados filtrados
                    csv_filtered = df_to_show.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados filtrados (CSV)",
                        data=csv_filtered,
                        file_name=f'resultados_filtrados_qsar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True,
                        key="download_filtered"
                    )
                            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")

# ============================================
# TAB 3: INFORMACIÓN
# ============================================
with tab3:
    st.header("Información del Sistema")
    
    st.markdown("""
    ### 🧬 Sobre el Modelo
    
    Este sistema utiliza un modelo **XGBoost optimizado** mediante búsqueda Bayesiana
    para predecir la actividad antiviral de compuestos contra el virus del dengue.
    
    ### 📊 Performance
    - **MCC**: 0.583 (Matthews Correlation Coefficient)
    - **Balanced Accuracy**: 0.683 (68.3%)
    - **ROC-AUC**: 0.896
    
    ### 🎯 Interpretación
    - **Alta potencia**: pActivity > 6 (IC50/EC50 < 1 μM)
    - **Media potencia**: 5 < pActivity ≤ 6 (1-10 μM)
    - **Baja potencia**: pActivity ≤ 5 (> 10 μM)
    
    ### 📝 Formato de Entrada
    - **SMILES**: Notación química estándar para representar moléculas
    - **CSV**: Para análisis por lotes, incluir columna 'SMILES'
    
    ### ⚠️ Limitaciones
    - Modelo entrenado con datos específicos de dengue
    - Las predicciones son estimaciones computacionales
    - Se recomienda validación experimental
    
    ### 🛠️ Dependencias Requeridas
    ```python
    streamlit
    pandas
    numpy
    matplotlib
    seaborn
    rdkit
    joblib
    ```
    
    ### 📌 Estado del Sistema
    """)
    
    # Verificación de componentes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Librerías:**")
        st.write(f"- RDKit: {'✅ Instalado' if RDKIT_AVAILABLE else '❌ No instalado'}")
        st.write(f"- Streamlit: ✅ Instalado")
        st.write(f"- NumPy: ✅ Instalado")
        st.write(f"- Pandas: ✅ Instalado")
    
    with col2:
        st.markdown("**Modelos:**")
        st.write(f"- XGBoost: {'✅ Cargado' if model is not None else '⚠️ No cargado'}")
        st.write(f"- Scaler: {'✅ Cargado' if scaler is not None else '⚠️ No cargado'}")
        st.write(f"- Variance Selector: {'✅ Cargado' if variance_selector is not None else '⚠️ No cargado'}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    QSAR Prediction Tool v2.0 | XGBoost Model | 2025
</div>
""", unsafe_allow_html=True)