from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from fpdf import FPDF
import hashlib

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage import img_as_ubyte
from scipy.stats import kurtosis, skew

# Initialize Flask app and directories
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
MASK_FOLDER = 'masks'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER
app.config['MASK_FOLDER'] = MASK_FOLDER

# Load the TFLite model
MODEL_PATH = "model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

###############################################################################
# 1) HELPER FUNCTIONS
###############################################################################
def get_seed_from_filename(filename):
    """Generate a deterministic seed based on the MD5 hash of the filename."""
    return int(hashlib.md5(filename.encode('utf-8')).hexdigest(), 16) % (2**32)

def add_deterministic_noise(image, noise_std=5, seed=42):
    """Add deterministic Gaussian noise to the image."""
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def preprocess_image(image):
    """Convert BGR to RGB and apply Gaussian blur."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def measure_infected_ratio(mask):
    """
    Measure how widespread the infected (white) region is inside the organ (gray) region.
    Assuming: 0->background, 1->organ, 2->infected.
    """
    organ_area = np.count_nonzero(mask > 0)
    infected_area = np.count_nonzero(mask == 2)
    if organ_area == 0:
        return 0.0
    return infected_area / organ_area

def infer_diagnosis(infected_ratio):
    """Risk scoring based solely on infected ratio."""
    if infected_ratio < 0.05:
        interpretation = "Normal tissue"
        risk_level = "Normal"
        recommendation = "No intervention required."
    elif infected_ratio < 0.10:
        interpretation = "Mild abnormalities detected"
        risk_level = "Mild Concern"
        recommendation = "Routine monitoring recommended."
    elif infected_ratio < 0.20:
        interpretation = "Moderate tissue irregularities"
        risk_level = "Intermediate Risk"
        recommendation = "Further clinical evaluation advised."
    elif infected_ratio < 0.40:
        interpretation = "Severe abnormalities detected"
        risk_level = "High Risk"
        recommendation = "Immediate further diagnostic tests required."
    else:
        interpretation = "Critical abnormalities detected"
        risk_level = "Critical Risk"
        recommendation = "Emergency intervention required."
    return {
        "risk_level": risk_level,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "risk_score": infected_ratio
    }

def get_base_filename(filename):
    """Return the filename without its extension."""
    return os.path.splitext(os.path.basename(filename))[0]

###############################################################################
# 2) RADIOMICS FEATURE EXTRACTION (GLCM + LBP) - For Reference Only
###############################################################################
def extract_radiomics_features(mask):
    """Extract GLCM and LBP texture features for academic reference."""
    if mask.ndim > 2:
        mask = mask[:, :, 0]
    mask_ubyte = img_as_ubyte(mask)
    glcm = graycomatrix(mask_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    lbp = local_binary_pattern(mask_ubyte, P=8, R=1, method='uniform')
    lbp_mean = float(np.mean(lbp))
    lbp_std = float(np.std(lbp))
    entropy_val = float(shannon_entropy(mask_ubyte))
    kurt_val = float(kurtosis(mask_ubyte.ravel()))
    skew_val = float(skew(mask_ubyte.ravel()))
    features = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'entropy': entropy_val,
        'kurtosis': kurt_val,
        'skewness': skew_val,
        'lbp_mean': lbp_mean,
        'lbp_std': lbp_std
    }
    return features

###############################################################################
# 3) VISUAL REPORTING FUNCTIONS
###############################################################################
def generate_3d_structure_plots(mask, base_filename, angles=[0, 30, 120]):
    """
    Generate 3D surface plots at different angles.
    Returns a list of filenames saved in REPORT_FOLDER.
    """
    plot_paths = []
    X, Y = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    for angle in angles:
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mask, cmap='viridis', edgecolor='none')
        ax.set_title(f'{angle}°', fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Intensity', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.view_init(elev=30, azim=angle)
        out_filename = f"3d_{angle}_{base_filename}.png"
        out_path = os.path.join(REPORT_FOLDER, out_filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(out_filename)
    return plot_paths

def generate_radiomics_radar_chart(radiomics_features, base_filename):
    """
    Generate a radar (spider) chart for selected radiomics features.
    Returns the filename of the generated radar chart.
    """
    import matplotlib.pyplot as plt
    features_to_plot = {
        'contrast': radiomics_features.get('contrast', 0),
        'homogeneity': radiomics_features.get('homogeneity', 0),
        'energy': radiomics_features.get('energy', 0),
        'entropy': radiomics_features.get('entropy', 0),
        'lbp_mean': radiomics_features.get('lbp_mean', 0),
        'kurtosis': radiomics_features.get('kurtosis', 0)
    }
    labels = list(features_to_plot.keys())
    values = list(features_to_plot.values())
    values += values[:1]
    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='b', linewidth=2)
    ax.fill(angles, values, color='b', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Radiomics Radar Chart", fontsize=10)
    radar_filename = f"radar_{base_filename}.png"
    radar_path = os.path.join(REPORT_FOLDER, radar_filename)
    plt.savefig(radar_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return radar_filename

def generate_overlay_image(original_image, mask, base_filename):
    """
    Generate and save an overlay image.
    Returns the filename of the overlay image.
    """
    mask_display = (mask * (255 / mask.max())).astype(np.uint8) if mask.max() > 0 else mask
    mask_color = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.7, mask_color, 0.3, 0)
    overlay_filename = f"overlay_{base_filename}.png"
    overlay_path = os.path.join(REPORT_FOLDER, overlay_filename)
    cv2.imwrite(overlay_path, overlay)
    return overlay_filename

def generate_enhanced_visual_report(mask, original_image, base_filename, decision, radiomics_features):
    """
    Generate a multi-panel PNG report (cover image for PDF).
    Returns the filename of the generated report.
    """
    report_filename = f"enhanced_report_{base_filename}.png"
    report_path = os.path.join(REPORT_FOLDER, report_filename)
    mask_display = (mask * (255 / mask.max())).astype(np.uint8) if mask.max() > 0 else mask
    mask_color = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_image, 0.7, mask_color, 0.3, 0)
    fig, axs = plt.subplots(3, 2, figsize=(14, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    axs[0, 0].imshow(overlay)
    axs[0, 0].set_title('Tissue Overlay')
    axs[0, 0].axis('off')
    axs[0, 1].imshow(mask, cmap='gray')
    axs[0, 1].set_title('Raw Generated Mask')
    axs[0, 1].axis('off')
    axs[1, 0].hist(mask.ravel(), bins=256, color='blue', alpha=0.7)
    axs[1, 0].set_title('Pixel Intensity Distribution')
    axs[1, 0].set_xlabel('Intensity')
    axs[1, 0].set_ylabel('Frequency')
    feature_text = "Radiomics Features (GLCM + LBP):\n"
    for k, v in radiomics_features.items():
        feature_text += f"{k}: {v:.3f}\n"
    axs[1, 1].text(0.5, 0.5, feature_text, fontsize=12, ha='center', va='center', color='green')
    axs[1, 1].set_title('Radiomics Feature Summary')
    axs[1, 1].axis('off')
    summary_text = (
        f"Diagnosis: {decision['interpretation']}\n"
        f"Risk Level: {decision['risk_level']}\n"
        f"Recommendation: {decision['recommendation']}\n"
        f"Infected Ratio (Risk Score): {decision['risk_score']:.3f}"
    )
    axs[2, 0].text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center', color='red')
    axs[2, 0].set_title('Diagnostic Summary')
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    axs[2, 1].set_title('Optional Panel')
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return report_filename

###############################################################################
# 4) PDF GENERATION
###############################################################################
class MediScanPDF(FPDF):
    def header(self):
        pass
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def generate_advanced_pdf_report(cover_image_path, three_d_image_paths, radar_image_path, decision, pdf_filename, radiomics_features):
    """
    Generate a multi-page PDF report.
    Returns the filename of the PDF.
    """
    pdf_path = os.path.join(REPORT_FOLDER, pdf_filename)
    pdf = MediScanPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", 'B', 16)
    # Page 1: Cover & Main Visual
    pdf.add_page()
    pdf.cell(190, 10, "MediScan AI - Advanced Image Analysis Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(190, 10, "This report provides a multi-angle 3D structural analysis, a radiomics radar chart, and an infected-area-based risk assessment.", align="C")
    pdf.ln(5)
    pdf.image(os.path.join(REPORT_FOLDER, cover_image_path), x=10, w=190)
    # Page 2: Detailed Analysis & Radiomics Table
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, "Detailed Analysis & Radiomics Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    diag_text = (
        f"Diagnosis: {decision['interpretation']}\n"
        f"Risk Level: {decision['risk_level']}\n"
        f"Recommendation: {decision['recommendation']}\n"
        f"Infected Ratio (Risk Score): {decision['risk_score']:.3f}\n\n"
        "Below are the radiomics features (GLCM + LBP) computed for reference. They provide deeper insight into texture patterns."
    )
    pdf.multi_cell(190, 10, diag_text)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Radiomics Feature", 1)
    pdf.cell(60, 10, "Value", 1, ln=True)
    pdf.set_font("Arial", '', 12)
    for k, v in radiomics_features.items():
        pdf.cell(60, 10, k, 1)
        pdf.cell(60, 10, f"{v:.3f}", 1, ln=True)
    # Page 3: 3D Multi-Angle Views & Radar Chart
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, "3D Structural Analysis & Radiomics Radar Chart", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(190, 10, "The following images depict the segmentation mask from different angles. The radar chart below innovatively visualizes key radiomics features.",)
    pdf.ln(5)
    for fname in three_d_image_paths:
        pdf.image(os.path.join(REPORT_FOLDER, fname), x=20, w=150)
        pdf.ln(5)
    pdf.image(os.path.join(REPORT_FOLDER, radar_image_path), x=20, w=150)
    pdf.ln(5)
    # Page 4: Extended Medical Notes & Recommendations
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 10, "Extended Medical Notes & Recommendations", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", '', 12)
    extended_notes = (
        "1. If the infected region exceeds 20% of the organ area, further evaluation with contrast-enhanced imaging or a biopsy is recommended.\n\n"
        "2. Patients with severe or critical infection (Infected Ratio > 40%) may require immediate intervention, including intravenous antibiotics, antiviral therapy, or surgical management.\n\n"
        "3. Elevated radiomics metrics, such as high contrast or skewness, may indicate heterogeneous lesion characteristics correlated with aggressive disease.\n\n"
        "4. Follow-up imaging in 2-4 weeks is advised for intermediate-risk cases to monitor progression.\n\n"
        "5. For academic research, these radiomics features can be correlated with histopathological findings to refine prognostic models."
    )
    pdf.multi_cell(190, 10, extended_notes)
    
    pdf.output(pdf_path)
    return pdf_filename

###############################################################################
# 5) MAIN PIPELINE
###############################################################################
def generate_final_reports(image_path):
    """
    Full processing pipeline:
      1) Read & preprocess image
      2) TFLite inference -> segmentation mask
      3) Save raw mask
      4) Compute radiomics features (GLCM + LBP) for reference
      5) Measure infected ratio & infer diagnosis
      6) Generate visual PNG report (cover image for PDF)
      7) Generate multiple 3D angle plots
      8) Generate a radiomics radar chart
      9) Generate multi-page PDF with advanced details
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image from path: {image_path}.")
    original_image = preprocess_image(image)
    input_height, input_width = input_details[0]['shape'][1:3]
    resized = cv2.resize(original_image, (input_width, input_height)).astype(np.float32) / 255.0
    resized = np.expand_dims(resized, axis=0)
    interpreter.set_tensor(input_details[0]['index'], resized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index']).copy()
    mask = np.argmax(output, axis=-1)[0].astype(np.uint8)
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    base_filename = get_base_filename(image_path)
    # Save raw mask with normalization for display purposes
    normalized_mask = (mask_resized.astype(np.float32) / (mask_resized.max() if mask_resized.max() > 0 else 1) * 255).astype(np.uint8)
    mask_filename = f"mask_{base_filename}.png"
    mask_path = os.path.join(app.config['MASK_FOLDER'], mask_filename)
    cv2.imwrite(mask_path, normalized_mask)
    radiomics_features = extract_radiomics_features(mask_resized)
    infected_ratio = measure_infected_ratio(mask_resized)
    decision = infer_diagnosis(infected_ratio)
    enhanced_visual_filename = generate_enhanced_visual_report(mask_resized, original_image, base_filename, decision, radiomics_features)
    three_d_filenames = generate_3d_structure_plots(mask_resized, base_filename, angles=[0, 30, 120])
    radar_filename = generate_radiomics_radar_chart(radiomics_features, base_filename)
    overlay_filename = generate_overlay_image(original_image, mask_resized, base_filename)
    pdf_filename = f"{base_filename}_enhanced.pdf"
    pdf_report_filename = generate_advanced_pdf_report(
        cover_image_path=enhanced_visual_filename,
        three_d_image_paths=three_d_filenames,
        radar_image_path=radar_filename,
        decision=decision,
        pdf_filename=pdf_filename,
        radiomics_features=radiomics_features
    )
    return {
        "pdf_report": pdf_report_filename,
        "overlay_image": overlay_filename,
        "raw_mask_image": mask_filename,
        "radar_chart_image": radar_filename,
        "three_d_images": three_d_filenames
    }, decision

###############################################################################
# 6) FLASK ROUTES
###############################################################################
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        vis_files, decision = generate_final_reports(filepath)
        return jsonify({
            "message": "Processing Complete",
            "report": vis_files["pdf_report"],
            "diagnosis": decision["interpretation"],
            "risk_level": decision["risk_level"],
            "recommendation": decision["recommendation"],
            "overlay_image": vis_files["overlay_image"],
            "raw_mask_image": vis_files["raw_mask_image"],
            "radar_chart_image": vis_files["radar_chart_image"],
            "three_d_images": vis_files["three_d_images"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reports/<filename>')
def serve_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)

@app.route('/masks/<filename>')
def serve_mask(filename):
    return send_from_directory(app.config['MASK_FOLDER'], filename)

@app.route('/visuals/<filename>')
def serve_visual(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
