# ⚽ Halı Saha Oyuncu Takip ve Analiz Sistemi

Futbol halı saha videolarında oyuncuların gerçek zamanlı tespiti, takibi ve hareket analizi için geliştirilmiş kapsamlı bir görüntü işleme ve yapay zeka projesi.

## 📋 Proje Hakkında

Bu proje, halı saha futbol videolarında oyuncuları tespit eder, her oyuncuya benzersiz ID atar ve hareketlerini kuş bakışı görünümde izler. Oyuncuların koşu mesafelerini metre cinsinden hesaplar ve segmentasyon ile hassas görselleştirme sağlar.

## 🎯 Ana Özellikler

### 1. 🔍 Oyuncu Tespiti ve Takibi

- **YOLOv8m** modeli ile yüksek doğrulukta insan tespiti
- **BoT-SORT** tracker ile stabil ID yönetimi
- Ani hareketlerde ve düşme-kalkma durumlarında ID stabilitesi
- Maksimum 14 oyuncu ID yönetimi

### 2. 🎨 İnteraktif Oyuncu Seçimi

- Matplotlib ile fare tıklama ile oyuncu seçimi
- Seçilen oyuncuya özel tracking
- İlk frame'de tüm oyuncuları gösterme
- Gerçek zamanlı seçim onayı

### 3. 🦅 Kuş Bakışı Görünümü (Bird's Eye View)

- **Perspektif Transformasyonu** ile top-down görünüm
- Gerçek saha koordinatları: 15m x 30m
- 4 nokta ile perspektif mapping:
  ```python
  SRC_POINTS: [[-80, 130], [450, 1900], [400, 100], [1230, 320]]
  DST_POINTS: [[50, 50], [50, 550], [350, 50], [350, 550]]
  ```
- OpenCV `cv2.getPerspectiveTransform()` ile dönüşüm matrisi

### 4. 📏 Mesafe Hesaplama

- Piksel-metre dönüşümü:
  - X ekseni: 20 px/m (300 piksel / 15m)
  - Y ekseni: 16.67 px/m (500 piksel / 30m)
- Gerçek zamanlı koşu mesafesi tracking
- Trajectory tabanlı toplam mesafe hesabı
- Metre cinsinden canlı görüntüleme

### 5. 🎭 Segmentasyon (MobileSAM)

- **MobileSAM** ile piksel düzeyinde oyuncu tespiti
- Bbox yerine tam silüet çıkarımı
- Renkli overlay ile görselleştirme (0.5 alpha)
- Contour çizimi ile hassas kenar belirleme
- Fallback mekanizması (SAM başarısız olursa bbox)

### 6. 📊 Görselleştirme Özellikleri

- Side-by-side video layout (orijinal + kuş bakışı)
- Renkli trajectory çizgileri (fade-out efekti)
- ID bazlı renk kodlaması (HSV color mapping)
- Kalın bbox ve label'lar
- Gerçek zamanlı istatistikler

## 🛠️ Kullanılan Teknolojiler

### Yapay Zeka ve Makine Öğrenmesi

- **Ultralytics YOLOv8**: Object detection
  - Model: YOLOv8m (medium) - denge (hız/doğruluk)
  - Class: Person (class 0)
  - Confidence threshold: 0.20-0.25
  - IOU threshold: 0.35-0.4
- **BoT-SORT Tracker**: Multi-object tracking

  - Built-in Ultralytics tracker
  - Camera motion compensation
  - Appearance feature matching
  - Robust to occlusions

- **MobileSAM**: Instance segmentation
  - Hafif SAM versiyonu
  - Bbox-prompted segmentation
  - Real-time capable

### Görüntü İşleme

- **OpenCV (cv2)**:

  - Perspektif transformasyonu (`getPerspectiveTransform`, `perspectiveTransform`)
  - Video okuma/yazma (`VideoCapture`, `VideoWriter`)
  - Çizim fonksiyonları (rectangle, circle, line, drawContours)
  - Renk dönüşümleri (HSV2BGR)
  - Contour detection (`findContours`)

- **NumPy**:
  - Array manipülasyonu
  - Matematiksel hesaplamalar
  - Mesafe hesaplamaları (Euclidean distance)
  - Mask işlemleri

### Görselleştirme ve UI

- **Matplotlib**:

  - İnteraktif oyuncu seçimi
  - Mouse event handling (`button_press_event`)
  - Frame görüntüleme
  - Interactive mode (plt.ion())

- **Collections**:
  - `deque`: Trajectory buffer (FIFO queue)
  - `defaultdict`: ID-based data structures

## 📐 Koordinat Sistemleri ve Dönüşümler

### 1. Video Koordinat Sistemi (Kaynak)

- Orijin: Sol üst köşe (0, 0)
- X ekseni: Sağa doğru artar
- Y ekseni: Aşağı doğru artar
- Çözünürlük: 1920x1080 (tipik)

### 2. Perspektif Transformasyonu

```python
# 4 kaynak nokta (video üzerinde saha köşeleri)
SRC_POINTS = [
    [-80, 130],    # Sol üst
    [450, 1900],   # Sol alt
    [400, 100],    # Sağ üst
    [1230, 320]    # Sağ alt
]

# 4 hedef nokta (kuş bakışı dikdörtgen)
DST_POINTS = [
    [50, 50],      # Sol üst
    [50, 550],     # Sol alt
    [350, 50],     # Sağ üst
    [350, 550]     # Sağ alt
]

# Dönüşüm matrisi
matrix = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

# Oyuncu pozisyonu dönüşümü
point = np.array([[foot_x, foot_y]], dtype=np.float32)
bird_pos = cv2.perspectiveTransform(point.reshape(-1, 1, 2), matrix)
```

### 3. Kuş Bakışı Koordinat Sistemi

- Boyut: 400x600 piksel
- Gerçek saha: 15m (genişlik) x 30m (yükseklik)
- Saha alanı: (50, 50) - (350, 550)
- Orta çizgi: y = 300
- Orta daire: merkez (200, 300), yarıçap 30px

### 4. Mesafe Hesaplama Formülü

```python
# Piksel-metre dönüşüm oranları
PIXELS_PER_METER_X = 300.0 / 15.0  # 20 px/m
PIXELS_PER_METER_Y = 500.0 / 30.0  # 16.67 px/m

# İki nokta arası mesafe
dx_meters = (x2 - x1) / PIXELS_PER_METER_X
dy_meters = (y2 - y1) / PIXELS_PER_METER_Y
distance = sqrt(dx_meters² + dy_meters²)

# Toplam koşu mesafesi
total_distance = Σ distance(point[i], point[i+1])
```

## 🎨 Renk Kodlama Sistemi

### ID Bazlı Renk Ataması

```python
# Her ID için benzersiz renk
hue = (track_id * 37) % 180  # HSV Hue değeri
color_hsv = [hue, 255, 255]  # Tam doygunluk ve parlaklık
color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)
```

### Renk Kullanım Alanları

- **Bbox**: Oyuncu çevresindeki dikdörtgen
- **Label**: ID metni arka planı
- **Trajectory**: Hareket çizgisi
- **Bird View**: Kuş bakışında nokta ve trajectory
- **SAM Mask**: Segmentasyon overlay

## 📁 Proje Yapısı

```
hali_saha/
│
├── hali_saha.ipynb          # Ana Jupyter notebook
├── Readme.md                # Bu dosya
├── hali.py                  # Python script versiyonu
│
├── shs2.mp4                 # Input video
│
├── Models/
│   ├── yolov8m.pt          # YOLO detection modeli
│   ├── yolov8n.pt          # YOLO nano modeli
│   ├── best_futbol.pt      # Custom player modeli
│   └── MobileSAM/
│       └── mobile_sam.pt   # Segmentation modeli
│
└── Outputs/
    ├── shs2_selected_person_tracking.mp4      # İnteraktif seçim + zoom
    ├── shs2_selected_tracking_bird.mp4        # İnteraktif seçim + orijinal
    └── shs2_selected_sam_tracking.mp4         # İnteraktif seçim + SAM
```

## 🚀 Kullanım

### Gereksinimler

```bash
pip install ultralytics opencv-python numpy matplotlib
```

### Notebook Hücreleri

#### 1. Temel Detection (Hücre 3-5)

Basit YOLOv8 ile person detection.

#### 2. İnteraktif Seçim + Zoom (Hücre 12)

- Fare ile oyuncu seçimi
- Seçilen oyuncuya zoom
- Kuş bakışı tracking

```python
OUTPUT: shs2_selected_person_tracking.mp4
```

#### 3. İnteraktif Seçim + Orijinal Video (Hücre 14)

- Orijinal videoda sadece seçilen oyuncu
- Diğer oyuncular ignore edilir
- Kuş bakışı tracking

```python
OUTPUT: shs2_selected_tracking_bird.mp4
```

#### 4. İnteraktif Seçim + SAM Segmentation (Hücre 16)

- MobileSAM ile segmentasyon
- Piksel seviyesinde hassasiyet
- Renkli overlay görselleştirme

```python
OUTPUT: shs2_selected_sam_tracking.mp4
```

## 📊 Performans ve Optimizasyon

### İşlem Hızları (RTX GPU)

- **YOLOv8m Detection**: ~30-40 FPS
- **BoT-SORT Tracking**: ~35-45 FPS
- **Perspektif Transform**: ~100+ FPS
- **SAM Segmentation**: ~15-20 FPS
- **Toplam (SAM ile)**: ~12-15 FPS

### Optimizasyon Teknikleri

1. **Trajectory Buffer**: `deque(maxlen=N)` ile bellek optimizasyonu
2. **Conditional Drawing**: Sadece stabil track'leri çiz (min_track_life)
3. **Alpha Blending**: Trajectory fade-out efekti
4. **Color Caching**: HSV-BGR dönüşümü tek seferlik
5. **Bbox Smoothing**: Titreşimi azaltmak için smoothing alpha

## 🎓 Teknik Detaylar

### BoT-SORT Tracker Parametreleri

```python
tracker="botsort.yaml"
persist=True           # ID'leri frame'ler arası koru
conf=0.25             # Detection confidence threshold
iou=0.4               # IoU threshold for NMS
```

### Trajectory Yönetimi

```python
TRAJECTORY_LEN = 60    # Son 60 nokta (2 saniye @ 30fps)
trajectories = defaultdict(lambda: deque(maxlen=TRAJECTORY_LEN))

# Alpha-based thickness
for i in range(len(points) - 1):
    alpha = (i + 1) / len(points)
    thickness = max(2, int(alpha * 7))
    cv2.line(frame, points[i], points[i+1], color, thickness)
```

### SAM Segmentation Pipeline

```python
# 1. Detection bbox'ını al
bbox = [x1, y1, x2, y2]

# 2. SAM çalıştır
sam_results = sam_model(frame, bboxes=[bbox])

# 3. Mask çıkar
mask = sam_results[0].masks.data[0].cpu().numpy()

# 4. Colored overlay oluştur
colored_mask = np.zeros_like(frame)
colored_mask[mask > 0.5] = color

# 5. Blend
cv2.addWeighted(colored_mask, 0.5, frame, 1.0, 0, frame)

# 6. Contour çiz
contours = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, color, 4)
```

## 🐛 Bilinen Sorunlar ve Çözümler

### 1. ID Değişimi Problemi

**Sorun**: Ani hareketlerde ID'ler değişiyor  
**Çözüm**: BoT-SORT tracker kullanımı + appearance features

### 2. Occlusion (Üst Üste Binme)

**Sorun**: Oyuncular üst üste bindiğinde tracking kayboluyor  
**Çözüm**: max_age parametresi ile hafıza tutma

### 3. Perspektif Distortion

**Sorun**: Saha kenarlarında mesafe hesabı hatalı  
**Çözüm**: Manuel SRC_POINTS ayarı + kalibrasyon

### 4. SAM Performance

**Sorun**: SAM ile işlem yavaşlıyor  
**Çözüm**: MobileSAM kullanımı + GPU acceleration

## 📈 Gelecek Geliştirmeler

- [ ] Otomatik saha köşe tespiti (Hough line transform)
- [ ] Çoklu oyuncu simultane tracking
- [ ] Heatmap görselleştirme
- [ ] Sprint speed analizi
- [ ] Takım bazlı ayırma (renk tespiti)
- [ ] Export to JSON (trajectory data)
- [ ] Real-time processing support

## 👨‍💻 Geliştirici

**HalilALPAK**  
Repository: [machine_learning](https://github.com/HalilALPAK/machine_learning)

## 📝 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.

---

**Son Güncelleme**: Kasım 2025  
**Versiyon**: 2.0
