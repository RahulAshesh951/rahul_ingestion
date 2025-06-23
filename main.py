import magic
import os
import subprocess
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import requests
import mimetypes
import re
import gc
import warnings
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification
import helpers
import easyocr
import tqdm as tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# ============================================================================
# FILE DETECTION AND SPLITTING (UNCHANGED)
# ============================================================================

def detect_file_type_with_fallback(file_path):
    try:
        mime_type = magic.from_file(file_path, mime=True)
        description = magic.from_file(file_path)
        print(f"🔍 File detection using magic: {mime_type} - {description}")

        if is_docsplit_supported(mime_type):
            print(f"✅ File type '{mime_type}' is supported by Docsplit")
            return mime_type, description
        else:
            print(f"⚠️ Magic detected '{mime_type}' but it's not supported by Docsplit, trying manual mapping...")

            extension = Path(file_path).suffix.lower()
            office_extension_mapping = {
                '.doc': 'application/msword',
                '.ppt': 'application/vnd.ms-powerpoint',
                '.xls': 'application/vnd.ms-excel',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xlsm': 'application/vnd.ms-excel.sheet.macroEnabled.12',
                '.pptm': 'application/vnd.ms-powerpoint.presentation.macroEnabled.12',
                '.docm': 'application/vnd.ms-word.document.macroEnabled.12',
                '.odt': 'application/vnd.oasis.opendocument.text',
                '.odp': 'application/vnd.oasis.opendocument.presentation',
                '.ods': 'application/vnd.oasis.opendocument.spreadsheet',
                '.rtf': 'application/rtf'
            }

            if extension in office_extension_mapping:
                manual_mime = office_extension_mapping[extension]
                print(f"✅ Manual mapping found: {extension} → {manual_mime}")
                return manual_mime, f"{description} (detected by manual extension mapping)"
            else:
                print(f"❌ Extension '{extension}' not found in manual mapping - file type not supported")
                return mime_type, description

    except Exception as e:
        print(f"❌ Error in file detection: {e}")
        try:
            fallback_mime, *_ = mimetypes.guess_type(file_path)
            if fallback_mime:
                print(f"🔄 Fallback detection using mimetypes: {fallback_mime}")
                return fallback_mime, f"Fallback detection (error in magic): {str(e)[:50]}"
        except:
            pass
        return None, None

def is_docsplit_supported(mime_type):
    supported_types = {
        'application/pdf', 'application/msword', 'application/vnd.ms-powerpoint',
        'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel.sheet.macroEnabled.12', 'application/vnd.ms-powerpoint.presentation.macroEnabled.12',
        'application/vnd.ms-word.document.macroEnabled.12', 'application/vnd.oasis.opendocument.text',
        'application/vnd.oasis.opendocument.presentation', 'application/vnd.oasis.opendocument.spreadsheet',
        'application/rtf', 'text/html', 'text/plain'
    }
    return mime_type in supported_types

def detect_file_type(file_path):
    return detect_file_type_with_fallback(file_path)

def check_docsplit_installation():
    try:
        result = subprocess.run(['docsplit', '--version'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def install_docsplit():
    try:
        subprocess.run(['gem', '--version'], check=True, capture_output=True)
        result = subprocess.run(['gem', 'install', 'docsplit'], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return True
        else:
            print(f"Failed to install Docsplit: {result.stderr}")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print("Error: Ruby/gem not found. Please install Ruby first.")
        return False

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(filename))]

def split_document(file_path, output_dir=None):
    if not check_docsplit_installation():
        print("Docsplit is not installed. Attempting installation...")
        if not install_docsplit():
            return False, []

    if output_dir is None:
        output_dir = f"{file_path}_pages"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        cmd = ['docsplit', 'images', '--format', 'png', '--size', '1024x', '--output', output_dir, file_path]
        print(f"📄 Running Docsplit command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        output_files = list(Path(output_dir).glob("*.png"))
        if output_files:
            sorted_files = sorted(output_files, key=natural_sort_key)
            print(f"✅ Found {len(sorted_files)} existing page images in {output_dir}:")
            for f in sorted_files:
                print(f"  - {f}")
            return True, [str(f) for f in sorted_files]

        if result.returncode == 0:
            output_files = list(Path(output_dir).glob("*.png"))
            if output_files:
                sorted_files = sorted(output_files, key=natural_sort_key)
                print(f"✅ Docsplit generated {len(sorted_files)} page images:")
                for f in sorted_files:
                    print(f"  - {f}")
                return True, [str(f) for f in sorted_files]
            else:
                print("❌ No output files found. Check the document format.")
                return False, []
        else:
            print(f"❌ Error splitting document: {result.stderr}")
            return False, []

    except subprocess.TimeoutExpired:
        print("❌ Error: Document splitting timed out")
        output_files = list(Path(output_dir).glob("*.png"))
        if output_files:
            sorted_files = sorted(output_files, key=natural_sort_key)
            print(f"✅ Found {len(sorted_files)} existing page images in {output_dir}:")
            for f in sorted_files:
                print(f"  - {f}")
            return True, [str(f) for f in sorted_files]
        return False, []
    except Exception as e:
        print(f"❌ Error during document splitting: {e}")
        output_files = list(Path(output_dir).glob("*.png"))
        if output_files:
            sorted_files = sorted(output_files, key=natural_sort_key)
            print(f"✅ Found {len(sorted_files)} existing page images in {output_dir}:")
            for f in sorted_files:
                print(f"  - {f}")
            return True, [str(f) for f in sorted_files]
        return False, []

# ============================================================================
# DESKEWING FUNCTIONS (UNCHANGED)
# ============================================================================

def deskew_image_robust(image: np.ndarray, debug: bool = False, debug_dir: str = "deskew_debug") -> tuple[np.ndarray, float]:
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "input.png"), image)

    current_image = image.copy()
    total_angle = 0.0

    if len(current_image.shape) == 3:
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = current_image.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "edges.png"), edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=50, maxLineGap=10)
    if lines is None:
        print("⚠️ No lines detected. Returning original image.")
        return image, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angles.append(angle)

    if not angles:
        print("⚠️ No valid angles detected. Returning original image.")
        return image, 0.0

    theta_median = np.median(angles)
    theta_normalized = round(theta_median / 90) * 90
    if theta_normalized > 180:
        theta_normalized -= 360
    elif theta_normalized <= -180:
        theta_normalized += 360

    is_inverted = abs(theta_normalized) == 180
    if is_inverted:
        theta_normalized = (theta_normalized + 180) % 360
        if theta_normalized > 180:
            theta_normalized -= 360
        print("✅ Detected inversion, adjusted angle by 180°")

    if abs(theta_normalized) > 0.1:
        (h, w) = current_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -theta_normalized, 1.0)
        current_image = cv2.warpAffine(
            current_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        total_angle = -theta_normalized
        if debug:
            cv2.imwrite(os.path.join(debug_dir, f"corrected.png"), current_image)

    return current_image, total_angle

def deskew_image_hough(image: np.ndarray, max_angle: float = 45.0, debug: bool = False, debug_dir: str = "deskew_debug") -> tuple[np.ndarray, float]:
    if debug:
        os.makedirs(debug_dir, exist_ok=True)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "edges_hough.png"), edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=50, maxLineGap=10)
    if lines is None:
        return image, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angle = angle % 90
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        return image, 0.0

    skew_angle = np.median(angles)
    skew_angle = np.clip(skew_angle, -max_angle, max_angle)

    if debug:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        deskewed = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        cv2.imwrite(os.path.join(debug_dir, "deskewed_hough.png"), deskewed)

    return image, skew_angle

def validate_deskew(image: np.ndarray, debug: bool = False, debug_dir: str = "deskew_debug", final: bool = False) -> float:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "validation_edges.png"), edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0 if not final else True

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        angle = angle % 90
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90
        angles.append(angle)

    if not angles:
        return 0.0 if not final else True

    angle_std = np.std(angles)
    if final:
        return angle_std < 5.0
    return angle_std

# ============================================================================
# YOLOv8x DETECTION FUNCTIONS (UNCHANGED)
# ============================================================================

def optimize_gpu_settings():
    if torch.cuda.is_available():
        print(f"🚀 GPU Optimization Settings:")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        print(f"   CuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   CuDNN Enabled: {torch.backends.cudnn.enabled}")
        return True
    return False

def download_pretrained_model():
    model_url = 'https://huggingface.co/hantian/yolo-doclaynet/resolve/main/yolov8x-doclaynet.pt'
    model_path = 'yolov8x-doclaynet.pt'
    if not os.path.exists(model_path):
        print(f"📥 Downloading YOLOv8x DocLayNet model...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            print(f"\n✅ Model downloaded successfully: {model_path}")
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            print("🔄 Using default YOLOv8x model instead...")
            model_path = 'yolov8x.pt'
    else:
        print(f"✅ Model already exists: {model_path}")
    return model_path

def load_yolo_model_optimized():
    model_path = download_pretrained_model()
    try:
        print(f"🔄 Loading YOLOv8x model with GPU optimizations...")
        model = YOLO(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model.to(device)
            print(f"✅ Model moved to GPU")
        print(f"✅ Model loaded successfully on {device}!")
        default_class_names = {
            0: 'Caption', 1: 'Footnote', 2: 'Formula', 3: 'List-item', 4: 'Page-footer',
            5: 'Page-header', 6: 'Picture', 7: 'Section-header', 8: 'Table', 9: 'Text', 10: 'Title'
        }
        class_names = model.names if hasattr(model, 'names') and model.names else default_class_names
        return model, class_names, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def validate_image(image):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.size == 0:
        print("❌ Empty image detected")
        return None
    if len(image.shape) < 2:
        print("❌ Invalid image dimensions")
        return None
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            pass
    h, w = image.shape[:2]
    if h < 32 or w < 32:
        print(f"❌ Image too small: {w}x{h}")
        return None
    if h > 4096 or w > 4096:
        print(f"⚠️ Large image detected: {w}x{h}, resizing...")
        scale = min(4096/w, 4096/h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def detect_layout_yolo_safe(images, model, device, imgsz=640, conf=0.25, iou=0.7, max_det=300):
    all_detections = []
    for i, image in enumerate(images):
        try:
            validated_img = validate_image(image)
            if validated_img is None:
                print(f"❌ Skipping invalid image {i+1}")
                all_detections.append([])
                continue
            print(f"🔍 Processing image {i+1} - Size: {validated_img.shape}")
            results = model.predict(
                source=validated_img,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
                save=False,
                show=False
            )
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        try:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            if x2 > x1 and y2 > y1 and confidence > 0:
                                detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(confidence),
                                    'class_id': class_id,
                                    'class_name': model.names.get(class_id, f'class_{class_id}')
                                })
                        except Exception as e:
                            print(f"⚠️ Error processing detection: {e}")
                            continue
            all_detections.append(detections)
            print(f"✅ Found {len(detections)} detections")
        except Exception as e:
            print(f"❌ Error processing image {i+1}: {e}")
            all_detections.append([])
            continue
        if device == 'cuda':
            torch.cuda.empty_cache()
    return all_detections

def gpu_memory_management():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"🧠 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def visualize_detections(image, detections, class_names, page_num=1, output_dir=None):
    fig, ax = plt.subplots(1, 1, figsize=(15, 20))
    ax.imshow(image)
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
              '#800080', '#FFA500', '#008000', '#808080', '#FFC0CB', '#8B4513',
              '#FF1493', '#00CED1', '#FF6347']
    print(f"\nPage {page_num} - YOLOv8 Detection Results:")
    print("-" * 65)
    class_counts = defaultdict(int)
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    for i, detection in enumerate(sorted_detections):
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        class_name = detection['class_name']
        class_counts[class_name] += 1
        color = colors[class_id % len(colors)]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        label_text = f"{class_name}\n{confidence:.3f}"
        ax.text(
            bbox[0], bbox[1] - 10, label_text, color='white',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
        )
        print(f"{i+1:2d}. {class_name:15s} | Conf: {confidence:.4f} | "
              f"Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
    ax.set_title(f"Page {page_num} - YOLOv8x Layout Detection ({len(detections)} elements)",
                 fontsize=16, pad=20, fontweight='bold')
    ax.axis('off')
    legend_elements = []
    for class_name, count in sorted(class_counts.items()):
        class_id = next((k for k, v in class_names.items() if v == class_name), 0)
        color = colors[class_id % len(colors)]
        legend_elements.append(patches.Patch(color=color, label=f'{class_name} ({count})'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1),
                  fontsize=10, framealpha=0.9)
    plt.tight_layout()

    if output_dir:
        output_path = os.path.join(output_dir, f"page_{page_num}_detection.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved visualization: {output_path}")
    else:
        plt.show()
        plt.close(fig)

    return class_counts

# ============================================================================
# ORIENTATION DETECTION WITH HEURISTIC METHOD
# ============================================================================

def ocr_elements_in_page_with_layout_order(crop_paths: list[dict], output_dir: str, debug: bool = False) -> dict:
    """
    Process pre-cropped elements with a heuristic method to detect orientation and rotate to left-to-right
    using PIL rotation for better quality.

    Args:
        crop_paths: List of dicts with crop paths, YOLO detection metadata, and LayoutLMv3 order.
        output_dir: Directory containing the crops.
        debug: If True, save intermediate crops for inspection.

    Returns:
        dict: Contains crop paths, orientations, orders, and errors (if any).
    """
    # Validate and convert output_dir to string
    if not isinstance(output_dir, (str, os.PathLike)):
        print(f"❌ Invalid output_dir type: {type(output_dir)}. Converting to string.")
        output_dir = str(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if debug:
        Path(output_dir + "_debug").mkdir(parents=True, exist_ok=True)

    if not crop_paths:
        print("❌ No crops found.")
        return {"error": "No crops", "results": []}

    results = []
    for idx, item in enumerate(tqdm.tqdm(crop_paths, desc="Processing elements"), start=1):
        crop_path = item["crop_path"]
        detection = item["detection"]
        order = item["order"]
        x1, y1, x2, y2 = map(int, detection["bbox"])
        class_name = detection["class_name"]
        print(f"[OCR] Element {idx}/{len(crop_paths)} ({class_name}, order {order}) at [x1={x1}, y1={y1}, x2={x2}, y2={y2}]", flush=True)

        try:
            crop_bgr = cv2.imread(crop_path)
            if crop_bgr is None:
                raise ValueError(f"Failed to load crop: {crop_path}")
            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            orig_height, orig_width = crop_pil.size
            # Optional resize only if smaller than 400x400
            if orig_height < 400 or orig_width < 400:
                scale = max(400 / orig_height, 400 / orig_width, 1.0)
                crop_pil = crop_pil.resize((int(orig_width * scale), int(orig_height * scale)), Image.Resampling.BICUBIC)
            if debug:
                cv2.imwrite(os.path.join(output_dir + "_debug", f"element_{order}_preprocessed.png"), cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"⚠️ Failed to process crop {idx}: {e}")
            continue

        orientation = "left-to-right"
        try:
            crop_gray = crop_pil.convert("L")
            pixels = list(crop_gray.getdata())
            height_pixels = sum(1 for i in range(0, len(pixels), orig_width) if pixels[i] < 250)
            width_pixels = sum(1 for i in range(orig_width) if pixels[i] < 250)
            if height_pixels > width_pixels * 1.5:
                orientation = "bottom-to-top"
                crop_pil = crop_pil.rotate(-90, expand=True)
                print(f"🔄 Heuristic detected vertical alignment, rotating element {idx} from 90° to 0°")
            if debug:
                cv2.imwrite(os.path.join(output_dir + "_debug", f"element_{order}_rotated.png"), cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"⚠️ Failed to process orientation for element {idx}: {e}")

        try:
            rotated_crop = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
            cv2.imwrite(crop_path, rotated_crop)
            results.append({
                "crop_path": crop_path,
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2],
                "orientation": orientation,
                "rotation_applied": -90 if orientation == "bottom-to-top" else 0,
                "order": order
            })
            print(f"✅ Saved {orientation} crop: {crop_path}")
        except Exception as e:
            print(f"⚠️ Failed to save element {idx}: {e}, skipping")
            continue

    print(f"✅ Processed {len(results)} crops in {output_dir}")
    return {"error": None, "results": results}


# ============================================================================
# INTEGRATION PIPELINE
# ============================================================================

def integration_pipeline(document_path: str, output_dir: str, debug: bool = False) -> dict:
    """
    Comprehensive pipeline to process a document:
    1. Detect file type and split into PNGs.
    2. Deskew using hough and robust methods.
    3. Detect elements with YOLO.
    4. Order elements with LayoutLMv3 using helpers.py (fallback to YOLO order if LayoutLMv3 fails).
    5. Crop elements in the determined order.
    6. Detect orientation with EasyOCR and save rotated crops.

    Args:
        document_path: Path to input document (image or multi-page file).
        output_dir: Directory to save outputs (pages, crops, visualizations).
        debug: If True, save intermediate outputs for debugging.

    Returns:
        dict: Results with page-wise crop paths, orders, and errors (if any).
    """
    # Step 1: File Type Detection and Splitting
    print(f"🔍 Detecting file type for {document_path}")
    extension = Path(document_path).suffix.lower()
    if extension in ['.jpg', '.jpeg', '.png']:
        print(f"✅ Image file detected: {extension}")
        page_paths = [document_path]
    else:
        print(f"📄 Non-image file detected, attempting to split: {extension}")
        success, page_paths = split_document(document_path, output_dir=os.path.join(output_dir, "pages"))
        if not success:
            print(f"❌ Failed to split document: {document_path}")
            return {"error": "Document splitting failed", "results": []}

    # Initialize YOLO model
    print("🚀 Initializing YOLO model")
    yolo_model, class_names, device = load_yolo_model_optimized()
    if yolo_model is None:
        print("❌ Failed to load YOLO model")
        return {"error": "YOLO model loading failed", "results": []}

    # Initialize LayoutLMv3 model (with fallback)
    print("🚀 Initializing LayoutLMv3 model")
    layoutlm_model = None
    try:
        layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
        layoutlm_model.eval()
        if device == 'cuda':
            layoutlm_model.to(device)
            print(f"✅ LayoutLMv3 model moved to GPU")
        print(f"✅ LayoutLMv3 model loaded successfully on {device}!")
    except Exception as e:
        print(f"⚠️ Failed to load LayoutLMv3 model: {e}. Continuing with YOLO detection order.")

    # Optimize GPU
    optimize_gpu_settings()

    all_results = []
    for i, page_path in enumerate(page_paths, 1):
        print(f"\n📄 Processing page {i}/{len(page_paths)}: {page_path}")

        # Step 2: Load and Deskew Image
        try:
            img = cv2.imread(page_path)
            if img is None:
                raise ValueError(f"Failed to load image: {page_path}")
            print(f"🖼️ Loaded image: {img.shape}")
        except Exception as e:
            print(f"❌ Failed to load page {i}: {e}")
            continue

        try:
            deskew_debug_dir = os.path.join(output_dir, f"page_{i}_deskew_debug")
            img_hough, skew_angle_hough = deskew_image_hough(img, debug=debug, debug_dir=deskew_debug_dir)
            print(f"✅ Deskewed page {i} (Hough) by {skew_angle_hough:.2f}°")
            if debug and abs(skew_angle_hough) > 0.1:
                cv2.imwrite(os.path.join(deskew_debug_dir, f"page_{i}_hough_deskewed.png"), img_hough)

            deskewed_img, skew_angle_robust = deskew_image_robust(img_hough, debug=debug, debug_dir=deskew_debug_dir)
            total_skew_angle = skew_angle_hough + skew_angle_robust
            print(f"✅ Deskewed page {i} (Robust) by {skew_angle_robust:.2f}°, total: {total_skew_angle:.2f}°")

            if not validate_deskew(deskewed_img, debug=debug, debug_dir=deskew_debug_dir, final=True):
                print(f"⚠️ Deskew validation failed for page {i}, using original image")
                deskewed_img = img
                total_skew_angle = 0.0
            if debug:
                cv2.imwrite(os.path.join(deskew_debug_dir, f"page_{i}_deskewed_final.png"), deskewed_img)
        except Exception as e:
            print(f"⚠️ Deskewing failed for page {i}: {e}, using original image")
            deskewed_img = img
            total_skew_angle = 0.0

        # Step 3: YOLO Layout Detection
        try:
            detections = detect_layout_yolo_safe([deskewed_img], yolo_model, device, imgsz=640, conf=0.25, iou=0.7, max_det=300)[0]
            print(f"✅ Detected {len(detections)} elements on page {i}")
        except Exception as e:
            print(f"❌ YOLO detection failed for page {i}: {e}")
            continue

        # Step 4: LayoutLMv3 Ordering (with fallback to YOLO order)
        ordered_detections = detections
        if detections and layoutlm_model is not None:
            try:
                # Normalize bounding boxes to 0-1000 scale for LayoutLMv3
                h, w = deskewed_img.shape[:2]
                boxes = [[int(x1*1000/w), int(y1*1000/h), int(x2*1000/w), int(y2*1000/h)] for x1, y1, x2, y2 in [d["bbox"] for d in detections]]
                inputs = helpers.boxes2inputs(boxes)
                inputs = helpers.prepare_inputs(inputs, layoutlm_model)
                with torch.no_grad():
                    outputs = layoutlm_model(**inputs)
                logits = outputs.logits.squeeze(0)  # [seq_len, num_labels]
                orders = helpers.parse_logits(logits, len(detections))
                if helpers.check_duplicate(orders):
                    print(f"⚠️ Duplicate orders detected, falling back to YOLO order")
                else:
                    ordered_detections = [detections[idx] for idx in orders]
                    print(f"✅ LayoutLMv3 ordered {len(ordered_detections)} elements")
            except Exception as e:
                print(f"⚠️ LayoutLMv3 ordering failed for page {i}: {e}, using YOLO order")

        # Step 5: Crop Elements in Determined Order
        try:
            crop_dir = os.path.join(output_dir, f"page_{i}_crops")
            Path(crop_dir).mkdir(parents=True, exist_ok=True)
            crop_paths = []
            img_pil = Image.fromarray(cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2RGB))
            for idx, d in enumerate(ordered_detections, 1):
                x1, y1, x2, y2 = map(int, d["bbox"])
                if x2 <= x1 or y2 <= y1:
                    print(f"⚠️ Invalid bounding box for element {idx}, skipping")
                    continue
                crop = img_pil.crop((x1, y1, x2, y2))
                crop_filename = f"element_{idx}_{d['class_name'].lower().replace(' ', '_')}.png"
                crop_path = os.path.join(crop_dir, crop_filename)
                crop.save(crop_path)
                crop_paths.append({"crop_path": crop_path, "detection": d, "order": idx})
                print(f"✅ Saved crop: {crop_path}")
        except Exception as e:
            print(f"❌ Cropping failed for page {i}: {e}")
            continue

        # Optional: Visualize Detections
        if debug:
            try:
                class_counts = visualize_detections(
                    img_pil,
                    ordered_detections,
                    class_names,
                    page_num=i,
                    output_dir=output_dir
                )
                print(f"✅ Visualized detections for page {i}: {dict(class_counts)}")
            except Exception as e:
                print(f"⚠️ Visualization failed for page {i}: {e}")

        # Step 6: EasyOCR Orientation Detection
        try:
            crop_result = ocr_elements_in_page_with_layout_order(
                crop_paths,
                crop_dir,
                debug=debug
            )
            if crop_result["error"]:
                print(f"❌ OCR processing failed for page {i}: {crop_result['error']}")
            else:
                print(f"✅ Processed {len(crop_result['results'])} elements for page {i}")
                all_results.append({"page": i, "path": page_path, "results": crop_result["results"]})
        except Exception as e:
            print(f"❌ OCR processing failed for page {i}: {e}")

        # Step 7: Memory Management
        gpu_memory_management()

    if not all_results:
        print("❌ No pages processed successfully")
        return {"error": "No pages processed", "results": []}

    print(f"✅ Pipeline completed: Processed {len(all_results)} pages")
    return {"error": None, "results": all_results}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    document_path = "/content/WhatsApp Image 2025-06-18 at 11.21.45_1e5c82d0.jpg"
    output_dir = "/content/output_crops"
    debug = True

    if not os.path.exists(document_path):
        print(f"❌ Image not found: {document_path}")
        return

    result = integration_pipeline(document_path, output_dir, debug)
    if result["error"]:
        print(f"❌ Pipeline failed: {result['error']}")
    else:
        print("✅ Pipeline completed successfully")
        for page_result in result["results"]:
            print(f"\n📄 Page {page_result['page']}: {page_result['path']}")
            for res in page_result["results"]:
                print(f"- Order {res['order']}: {res['class_name']} at {res['bbox']}: {res['orientation']} (rotated {res['rotation_applied']}°), saved at {res['crop_path']}")

if __name__ == "__main__":
    main()